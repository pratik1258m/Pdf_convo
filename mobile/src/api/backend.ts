import type {
  AskTextRequest,
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

function httpDetail(payload: unknown, response: { status: number; statusText: string }): string {
  const detail = detailFromPayload(payload)
  return detail ?? `${response.status} ${response.statusText}`
}

function detailFromPayload(payload: unknown): string | null {
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
  return null
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return ''
}

export function getHealth(baseUrl: string, signal?: AbortSignal): Promise<HealthSnapshot> {
  return request<HealthSnapshot>(baseUrl, '/api/health', { timeoutMs: 8000, signal })
}

export function setApiKey(
  baseUrl: string,
  apiKey: string,
  provider: string = 'groq',
  model?: string
): Promise<{ ok: boolean; engine?: string; message: string }> {
  return request<{ ok: boolean; engine?: string; message: string }>(baseUrl, '/api/set_api_key', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey, provider, model }),
    timeoutMs: 15000,
  })
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
  input: AskTextRequest,
  signal?: AbortSignal
): Promise<AskTextResponse> {
  // An exchange-less body leaves `history` out rather than sending an empty list:
  // absent means "nothing to expand or re-ask", which a list of nothing does not say.
  const { history, ...rest } = input
  const body: AskTextRequest = history && history.length ? { ...rest, history } : rest

  return request<AskTextResponse>(
    baseUrl,
    '/api/ask_text',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      timeoutMs: 30000,
      signal,
    }
  )
}

/** Speech is a separate, cache-first call so the answer is already on screen. */
const SPEAK_BATCH = 12

export async function speakSegments(
  baseUrl: string,
  texts: string[],
  voice: string,
  signal?: AbortSignal
): Promise<SpeakResponse> {
  // A local speech engine makes one clip at a time, so twenty segments in a single
  // request would hold the whole queue behind them and run past the client's clock.
  // Batches go out one after another, and a batch that fails after the first has
  // landed still hands back the audio that exists - a shorter reply, not a lost one.
  const batches: string[][] = []
  for (let index = 0; index < texts.length; index += SPEAK_BATCH) batches.push(texts.slice(index, index + SPEAK_BATCH))
  if (!batches.length) batches.push([])

  const parts: SpeakResponse[] = []
  for (const batch of batches) {
    try {
      parts.push(await request<SpeakResponse>(baseUrl, '/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: batch, voice }),
        timeoutMs: 60000,
        signal,
      }))
    } catch (error) {
      if (signal?.aborted || !parts.length) throw error
      break
    }
  }

  if (parts.length === 1) return parts[0]
  const segments = parts.flatMap((part) => part.segments)
  return {
    ...parts[0],
    status: parts.some((part) => part.status !== 'success') || segments.length < texts.length ? 'partial' : parts[0].status,
    segments,
    requested: parts.reduce((total, part) => total + (part.requested ?? 0), 0),
    errors: parts.flatMap((part) => part.errors ?? []),
    timings: {
      tts_ms: parts.reduce((total, part) => total + (part.timings?.tts_ms ?? 0), 0),
      segments: segments.length,
      cached: parts.reduce((total, part) => total + (part.timings?.cached ?? 0), 0),
    },
  }
}

type SelectedFile = { uri: string; name: string; mimeType?: string | null }

/**
 * The document goes up over XMLHttpRequest, not fetch.
 *
 * React Native's `FormData` is an XHR polyfill: a `{uri, name, type}` part is only
 * serialised when XHR builds the request body, and `fetch` throws
 * "Unsupported FormDataPart implementation" on that same object - which is exactly how
 * the first on-device upload died. The picker copies the file into the app's own cache
 * directory, so the URI it hands back is a path the native uploader can open.
 */
export function uploadPdf(baseUrl: string, file: SelectedFile, signal?: AbortSignal): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', {
    uri: file.uri,
    name: file.name,
    type: file.mimeType || 'application/pdf',
  } as unknown as string)

  return new Promise<UploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${baseUrl}/api/upload_pdf`)
    xhr.timeout = 120000
    xhr.responseType = 'text'

    const onAbort = () => xhr.abort()
    if (signal) signal.addEventListener('abort', onAbort)
    const settle = () => {
      if (signal) signal.removeEventListener('abort', onAbort)
    }

    xhr.onload = () => {
      settle()
      let payload: unknown = null
      if (xhr.responseText) {
        try {
          payload = JSON.parse(xhr.responseText)
        } catch {
          payload = null
        }
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        if (payload === null) {
          reject(new BackendError('parse', 'The server sent an empty response.', xhr.status))
        } else {
          resolve(payload as UploadResponse)
        }
        return
      }
      reject(
        new BackendError(
          'http',
          detailFromPayload(payload) ?? `${xhr.status} ${xhr.statusText}`,
          xhr.status
        )
      )
    }
    xhr.onerror = () => {
      settle()
      reject(new BackendError('network', 'Could not send the document. Check that the server is running.'))
    }
    xhr.ontimeout = () => {
      settle()
      reject(new BackendError('timeout', 'The server did not finish indexing the document within 2 minutes.'))
    }
    xhr.onabort = () => {
      settle()
      reject(new BackendError('timeout', 'The upload was cancelled.'))
    }
    xhr.send(form)
  })
}
