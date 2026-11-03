import type {
  AskTextResponse,
  DocumentSnapshot,
  HealthSnapshot,
  SpeakResponse,
  UploadResponse,
  VoiceOption,
} from '../types'

export type BackendErrorKind = 'network' | 'timeout' | 'http' | 'parse' | 'unreachable'

export class BackendError extends Error {
  kind: BackendErrorKind
  status?: number

  constructor(kind: BackendErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'BackendError'
    this.kind = kind
    this.status = status
  }
}

type FetchInit = {
  method?: string
  body?: BodyInit | null
  headers?: Record<string, string>
  timeoutMs?: number
  signal?: AbortSignal
}

async function request<T>(baseUrl: string, path: string, init: FetchInit = {}): Promise<T> {
  const { timeoutMs = 15000, ...rest } = init
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  if (rest.signal) {
    rest.signal.addEventListener('abort', () => controller.abort())
  }

  let response: Response
  try {
    response = await fetch(`${baseUrl}${path}`, { ...rest, signal: controller.signal })
  } catch (error) {
    clearTimeout(timer)
    const aborted = error instanceof Error && error.name === 'AbortError'
    throw new BackendError(
      aborted ? 'timeout' : 'network',
      aborted
        ? `The server did not answer within ${Math.round(timeoutMs / 1000)}s.`
        : errorMessage(error) || 'Could not reach the server. Check that it is running and on the same Wi-Fi.'
    )
  }
  clearTimeout(timer)

  let payload: unknown = null
  const text = await response.text()
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      if (!response.ok) {
        throw new BackendError('http', `${response.status} ${response.statusText}: ${text.slice(0, 160)}`, response.status)
      }
      throw new BackendError('parse', 'The server replied with something that is not JSON.', response.status)
    }
  }

  if (!response.ok) {
    throw new BackendError('http', httpDetail(payload, response), response.status)
  }
  if (payload === null) throw new BackendError('parse', 'The server sent an empty response.')
  return payload as T
}

function httpDetail(payload: unknown, response: Response): string {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      const first = detail[0]
      if (first && typeof first === 'object' && 'msg' in first) {
        return String((first as { msg: unknown }).msg).replace(/^Value error,\s*/i, '')
      }
    }
  }
  return `${response.status} ${response.statusText}`
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return ''
}

export function getHealth(baseUrl: string, signal?: AbortSignal): Promise<HealthSnapshot> {
  return request<HealthSnapshot>(baseUrl, '/api/health', { timeoutMs: 8000, signal })
}

export function getVoices(baseUrl: string): Promise<VoiceOption[]> {
  return request<{ voices: VoiceOption[] }>(baseUrl, '/api/voices', { timeoutMs: 10000 }).then(
    (data) => data.voices ?? []
  )
}

export function getDocument(baseUrl: string): Promise<DocumentSnapshot> {
  return request<DocumentSnapshot>(baseUrl, '/api/document', { timeoutMs: 8000 })
}

/** Text first: retrieval and answer synthesis only, so speech never blocks the screen. */
export function askText(
  baseUrl: string,
  input: { query: string; voice?: string },
  signal?: AbortSignal
): Promise<AskTextResponse> {
  return request<AskTextResponse>(
    baseUrl,
    '/api/ask_text',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
      timeoutMs: 20000,
      signal,
    }
  )
}

/** Speech is a separate, cache-first call so the answer is already on screen. */
export function speakSegments(
  baseUrl: string,
  texts: string[],
  voice: string,
  signal?: AbortSignal
): Promise<SpeakResponse> {
  return request<SpeakResponse>(
    baseUrl,
    '/api/speak',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, voice }),
      timeoutMs: 60000,
      signal,
    }
  )
}

type SelectedFile = { uri: string; name: string; mimeType?: string | null }

export function uploadPdf(baseUrl: string, file: SelectedFile, signal?: AbortSignal): Promise<UploadResponse> {
  const form = new FormData()
  // React Native's FormData serialises a {uri, name, type} object as a file part;
  // the Content-Type header must stay unset so the runtime can add the multipart boundary.
  form.append('file', {
    uri: file.uri,
    name: file.name,
    type: file.mimeType || 'application/pdf',
  } as unknown as Blob)

  return request<UploadResponse>(
    baseUrl,
    '/api/upload_pdf',
    { method: 'POST', body: form, timeoutMs: 120000, signal }
  )
}
