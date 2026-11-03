export function timeLabel(epochMs: number): string {
  const date = new Date(epochMs)
  const hours = date.getHours()
  const suffix = hours >= 12 ? 'PM' : 'AM'
  const hour12 = ((hours + 11) % 12) + 1
  return `${hour12}:${String(date.getMinutes()).padStart(2, '0')} ${suffix}`
}

export function formatLatency(ms?: number): string | null {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return null
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`
  // Retrieval really is under a millisecond, so rounding it to "0 ms" would be a lie.
  if (ms < 10) return `${ms.toFixed(1)} ms`
  return `${Math.round(ms)} ms`
}

/** `hi_IN-pratham-medium` -> `hi-IN`, `en-IN-PrabhatNeural` -> `en-IN`.
 *  Anything that is not shaped like a locale (`supertonic-M1`) gets the default
 *  rather than a string the recognizer would reject. */
export function localeForVoice(voiceId: string): string {
  const tokens = voiceId.replace(/_/g, '-').split('-')
  const language = tokens[0] ?? ''
  const region = tokens[1] ?? ''
  if (/^[a-z]{2}$/.test(language) && /^[A-Za-z]{2}$/.test(region)) {
    return `${language}-${region.toUpperCase()}`
  }
  return 'en-IN'
}

/** The language to listen in. A local voice id is not shaped like an Edge one,
 *  so the server's own locale label wins and the id is only a guess of last resort. */
export function micLocale(voice: { locale?: string } | undefined, voiceId: string): string {
  return voice?.locale || localeForVoice(voiceId)
}

export type ListenMode = 'auto' | 'en-IN' | 'hi-IN'

/** `auto` follows the reply voice. An explicit choice wins, because the server's
 *  one Hindi-and-English voice can be asked to listen in either language. */
export function resolveMicLocale(
  listen: ListenMode,
  voice: { locale?: string } | undefined,
  voiceId: string
): string {
  return listen === 'auto' ? micLocale(voice, voiceId) : listen
}

export function shortHost(baseUrl: string): string {
  return baseUrl.replace(/^https?:\/\//i, '') || 'not connected'
}

export function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`
}
