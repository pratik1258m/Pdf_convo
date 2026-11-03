import { useCallback, useEffect, useRef, useState } from 'react'
import { setAudioModeAsync, useAudioPlayer, useAudioPlayerStatus } from 'expo-audio'
import { resolveAudioUrl } from '../api/normalize'

type Session = { id: string; urls: string[]; index: number; stopped: boolean; skipped: number }

/**
 * Plays an answer as a queue of sentence-sized clips.
 *
 * The server returns the first clip while later ones are still being
 * synthesised, so `playSegments` starts speech immediately and `enqueue`
 * extends the same session when the rest land. React Native cannot stream a
 * response body, which is why the head and the tail are separate requests.
 */
export function useReplyPlayer(baseUrl: string, muted: boolean) {
  const player = useAudioPlayer(null, { updateInterval: 200 })
  const status = useAudioPlayerStatus(player)
  const sessionRef = useRef<Session | null>(null)
  const activeRef = useRef<string | null>(null)
  const advancedRef = useRef(false)
  const errorRef = useRef(false)
  const [activeId, setActiveIdState] = useState<string | null>(null)
  const [segment, setSegment] = useState(0)
  const [playbackError, setPlaybackError] = useState<string | null>(null)

  useEffect(() => {
    void setAudioModeAsync({
      allowsRecording: false,
      playsInSilentMode: true,
      shouldRouteThroughEarpiece: false,
      interruptionMode: 'doNotMix',
    })
  }, [])

  useEffect(() => {
    player.muted = muted
  }, [muted, player])

  const markActive = useCallback((id: string | null) => {
    activeRef.current = id
    setActiveIdState(id)
  }, [])

  const load = useCallback(
    (index: number) => {
      const session = sessionRef.current
      if (!session) {
        markActive(null)
        return
      }
      const uri = session.urls[index]
      if (!uri) {
        // Ran out of clips. Keep the session so a late `enqueue` can resume it.
        session.index = session.urls.length
        markActive(null)
        // Clips that were skipped are reported once, when the queue is over.
        if (session.skipped) {
          setPlaybackError(
            `${session.skipped} of ${session.urls.length} clips could not be played. The rest of that answer is on screen.`
          )
        }
        return
      }
      session.index = index
      advancedRef.current = false
      errorRef.current = false
      markActive(session.id)
      setSegment(index)
      player.replace({ uri })
      player.play()
    },
    [markActive, player]
  )

  useEffect(() => {
    if (!status.didJustFinish) {
      advancedRef.current = false
      return
    }
    // expo-audio keeps `didJustFinish` true until the next clip starts, so the
    // advance must run once per finished clip or the queue would skip one.
    if (advancedRef.current) return
    advancedRef.current = true
    const session = sessionRef.current
    if (session?.stopped) return
    load(session ? session.index + 1 : 0)
  }, [status.didJustFinish, load])

  useEffect(() => {
    if (!status.error) {
      errorRef.current = false
      return
    }
    // The same failure stays on the status until the next clip loads, so it has
    // to be counted once or one dead clip would skip the whole answer.
    if (errorRef.current) return
    errorRef.current = true
    const session = sessionRef.current
    if (!session || session.stopped || !session.urls.length) {
      sessionRef.current = null
      markActive(null)
      setPlaybackError(`Audio could not be played: ${formatError(status.error)}`)
      return
    }
    // One dead clip - an evicted file, a connection dropped mid-download - must
    // not silence the rest of the answer. Move on and report it when the queue ends.
    session.skipped += 1
    load(session.index + 1 < session.urls.length ? session.index + 1 : session.urls.length)
  }, [load, markActive, status.error])

  const playSegments = useCallback(
    (paths?: string[] | null, id?: string) => {
      const uris = resolveAll(baseUrl, paths)
      if (!uris.length) {
        setPlaybackError('The server returned this answer without audio.')
        return
      }
      setPlaybackError(null)
      sessionRef.current = { id: id ?? uris[0], urls: uris, index: 0, stopped: false, skipped: 0 }
      setSegment(0)
      load(0)
    },
    [baseUrl, load]
  )

  /** Append the clips that arrive after playback already started. */
  const enqueue = useCallback(
    (paths?: string[] | null, id?: string) => {
      const session = sessionRef.current
      const uris = resolveAll(baseUrl, paths)
      if (!session || !uris.length || (id && session.id !== id)) return
      session.urls = [...session.urls, ...uris]
      setSegment(session.index)
      if (session.stopped || activeRef.current === session.id) return
      load(session.index)
    },
    [baseUrl, load]
  )

  const stop = useCallback(() => {
    const session = sessionRef.current
    if (session) session.stopped = true
    player.pause()
    setSegment(0)
    markActive(null)
  }, [player, markActive])

  return {
    playSegments,
    enqueue,
    stop,
    activeId,
    segment,
    isPlaying: activeId !== null && status.playing,
    error: playbackError,
  }
}

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message
  const text = String(error)
  return text.length > 140 ? `${text.slice(0, 140)}…` : text
}

function resolveAll(baseUrl: string, paths?: string[] | null): string[] {
  return (paths ?? [])
    .map((path) => resolveAudioUrl(baseUrl, path))
    .filter((uri): uri is string => Boolean(uri))
}
