export type VoiceOption = {
  id: string
  name: string
  locale: string
  gender: string
  recommended: boolean
  /** Which engine speaks this voice: `piper` and `supertonic` run on the server
   *  machine, `edge` is a network call. Decides how long the first sentence takes. */
  engine?: string
  format?: string
  local?: boolean
  /** False when this machine cannot make audio for the voice, e.g. the model
   *  file was never downloaded. Such a voice is listed but never selectable. */
  available?: boolean
  reason?: string
  languages?: string
}

export type ChunkingProfile = {
  chunks: number
  words_min: number
  words_p50: number
  words_max: number
  words_avg: number
  target_words: number
  headings_tracked: number
  short_runs_merged: number
  pages_with_chunks: number
  unique_terms: number
  text_preserved_pct: number
}

export type DocumentStats = {
  filename: string
  total_pages: number
  total_chunks: number
  summary: string
  index_ms?: number
  chunking?: Partial<ChunkingProfile>
}

export type HealthSnapshot = {
  status: string
  current_document: string
  total_chunks: number
  available_voices: number
  /** The voice the server would use if this phone expressed no preference. */
  default_voice?: string
  /** Per-engine availability, so Settings can say why the fast voice is greyed out. */
  speech?: Record<string, { local: boolean; format: string; voices: number; usable: number; available: boolean; why_not: string }>
  engine?: string
  chunking?: Partial<ChunkingProfile>
}

export type DocumentSnapshot = { loaded: boolean } & Partial<DocumentStats>

/** Server-side stage times. Absent until the stage has run. */
export type Timings = {
  retrieval_ms?: number
  answer_ms?: number
  total_ms?: number
  tts_ms?: number
}

/** One completed exchange, older first.
 *  Snake_case because that is the shape `QueryRequest.history` in server.py reads,
 *  and `query` is what the follow-up expansion and the correction re-ask look at. */
export type AskHistoryTurn = {
  query: string
  answer: string
}

export type AskTextRequest = {
  query: string
  voice?: string
  /** Omitted entirely when there is nothing completed to send. */
  history?: AskHistoryTurn[]
}

export type AskTextResponse = {
  status: string
  query: string
  answer: string
  spoken_text: string
  spoken_segments: string[]
  is_out_of_context: boolean
  /** True when nothing is indexed at all, which is not the same as nothing found. */
  no_document?: boolean
  sources: number[]
  retrieval_score: number
  confidence_score: number
  engine: string
  timings: Timings
}

export type SpokenSegment = {
  text: string
  filename: string
  audio_url: string
  cached: boolean
  byte_size: number
}

export type SpeakResponse = {
  /** `partial` means some of what was asked for could not be turned into audio. */
  status: string
  voice: string
  /** Which engine made these clips, so a 0.1 s answer and a 2 s answer are explainable. */
  speech_engine?: string
  segments: SpokenSegment[]
  requested?: number
  errors?: (string | null)[]
  timings: { tts_ms: number; segments: number; cached: number }
}

export type UploadResponse = {
  status: string
  stats: DocumentStats
  welcome_text: string
  welcome_segments: string[]
  engine?: string
}

export type ChatRole = 'user' | 'assistant' | 'system'

/** `working` = no text yet, `voice` = text on screen, speech still being made. */
export type AssistantStage = 'working' | 'voice' | 'idle'

export type ChatMessage = {
  id: string
  role: ChatRole
  text: string
  createdAt: number
  sources?: number[]
  outOfContext?: boolean
  noDocument?: boolean
  audioUrls?: string[] | null
  voice?: string
  latencyMs?: number
  stage?: AssistantStage
  failed?: boolean
  engine?: string
  score?: number
  timings?: Timings
}
