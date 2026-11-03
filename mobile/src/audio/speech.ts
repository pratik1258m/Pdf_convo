import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
} from 'expo-speech-recognition'

export type SpeechPhase = 'idle' | 'requesting' | 'listening' | 'stopping'

export type SpeechErrorInfo = { code: string; message: string }

export type SpeechAvailability = {
  available: boolean
  onDevice: boolean
  detail: string | null
}

// The native volumechange event reports a level between -2 and 10 where <= 0 is inaudible.
function normalizeLevel(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, (value + 2) / 12))
}

const ERROR_COPY: Record<string, string> = {
  'no-speech': 'No speech detected. Hold the mic and speak for a second longer.',
  network: 'Recognition needs internet, or switch to an on-device language in Android settings.',
  'not-allowed': 'Microphone or speech permission was denied. Enable it in Settings.',
  'service-not-allowed': 'Speech recognition permission was denied. Enable it in Settings.',
  'audio-capture': 'The microphone is in use by another app or by the keyboard.',
  'language-not-supported': 'This language is not installed on the device. Download it or pick another voice.',
  busy: 'The recognizer is still finishing the previous phrase. Try again.',
  interrupted: 'Recognition was interrupted by a call or another app.',
  aborted: 'Recognition was cancelled.',
  'speech-timeout': 'You stopped talking before a phrase was recognised.',
  'too-many-requests': 'Recognition rate limit reached. Wait a moment and retry.',
  client: 'The recognizer failed. Tap the mic again.',
  unknown: 'Recognition failed for an unknown reason.',
}

export function describeSpeechError(code: string, rawMessage?: string): string {
  const friendly = ERROR_COPY[code]
  if (friendly) return rawMessage ? `${friendly} (${rawMessage})` : friendly
  return rawMessage || 'Recognition failed.'
}

export function useSpeechRecognition(options: {
  locale: string
  onTranscript: (text: string, isFinal: boolean) => void
}) {
  const { locale, onTranscript } = options
  const [phase, setPhase] = useState<SpeechPhase>('idle')
  const [interim, setInterim] = useState('')
  const [level, setLevel] = useState(0)
  const [speechError, setSpeechError] = useState<SpeechErrorInfo | null>(null)

  const interimRef = useRef('')
  const settledRef = useRef(true)
  const safetyTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const transcriptRef = useRef(onTranscript)

  useEffect(() => {
    transcriptRef.current = onTranscript
  }, [onTranscript])

  const clearSafetyTimer = useCallback(() => {
    if (safetyTimer.current) {
      clearTimeout(safetyTimer.current)
      safetyTimer.current = null
    }
  }, [])

  const settle = useCallback((text: string | null, isFinal: boolean) => {
    if (settledRef.current) return
    settledRef.current = true
    clearSafetyTimer()
    setPhase('idle')
    setLevel(0)

    const captured = (text ?? interimRef.current).trim()
    interimRef.current = ''
    setInterim('')
    if (captured) transcriptRef.current(captured, isFinal)
  }, [clearSafetyTimer])

  useSpeechRecognitionEvent('start', () => {
    setSpeechError(null)
    setPhase('listening')
  })

  useSpeechRecognitionEvent('result', (event) => {
    const text = event.results?.[0]?.transcript?.trim() ?? ''
    if (event.isFinal) {
      settle(text, true)
      return
    }
    if (text) {
      interimRef.current = text
      setInterim(text)
    }
  })

  useSpeechRecognitionEvent('error', (event) => {
    setSpeechError({ code: event.error, message: describeSpeechError(event.error, event.message) })
    // A failed phrase must never be sent as if it were a final transcript, but anything
    // already heard is handed back as a draft the user can edit and send by hand.
    if (!settledRef.current) {
      settledRef.current = true
      clearSafetyTimer()
      setPhase('idle')
      setLevel(0)
      const captured = interimRef.current.trim()
      interimRef.current = ''
      setInterim('')
      if (captured) transcriptRef.current(captured, false)
    }
  })

  useSpeechRecognitionEvent('end', () => {
    settle(null, false)
  })

  useSpeechRecognitionEvent('volumechange', (event) => {
    setLevel(normalizeLevel(event.value))
  })

  const availability = useMemo<SpeechAvailability>(() => {
    try {
      return {
        available: ExpoSpeechRecognitionModule.isRecognitionAvailable(),
        onDevice: ExpoSpeechRecognitionModule.supportsOnDeviceRecognition(),
        detail: null,
      }
    } catch (error) {
      return {
        available: false,
        onDevice: false,
        detail: error instanceof Error ? error.message : 'Speech recognizer is not present in this build.',
      }
    }
  }, [])

  const start = useCallback(async () => {
    if (phase !== 'idle') return false
    setSpeechError(null)
    setPhase('requesting')
    try {
      const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync()
      if (!permission.granted) {
        setPhase('idle')
        setSpeechError({
          code: 'not-allowed',
          message: describeSpeechError('not-allowed', permission.canAskAgain ? undefined : 'Do not ask again is set'),
        })
        return false
      }
      settledRef.current = false
      clearSafetyTimer()
      interimRef.current = ''
      setInterim('')
      ExpoSpeechRecognitionModule.start({
        lang: locale,
        interimResults: true,
        continuous: false,
        maxAlternatives: 1,
        addsPunctuation: true,
        volumeChangeEventOptions: { enabled: true, intervalMillis: 120 },
        iosTaskHint: 'dictation',
        iosCategory: {
          category: 'playAndRecord',
          categoryOptions: ['defaultToSpeaker', 'allowBluetooth'],
          mode: 'default',
        },
      })
      return true
    } catch (error) {
      settledRef.current = true
      setPhase('idle')
      setSpeechError({
        code: 'client',
        message: error instanceof Error ? error.message : 'Could not start the microphone.',
      })
      return false
    }
  }, [clearSafetyTimer, locale, phase])

  const stop = useCallback(() => {
    if (phase !== 'listening' && phase !== 'requesting') return
    setPhase('stopping')
    try {
      ExpoSpeechRecognitionModule.stop()
    } catch {
      settle(null, false)
      return
    }
    clearSafetyTimer()
    safetyTimer.current = setTimeout(() => settle(null, false), 3000)
  }, [clearSafetyTimer, phase, settle])

  const abort = useCallback(() => {
    settledRef.current = true
    clearSafetyTimer()
    interimRef.current = ''
    setInterim('')
    setPhase('idle')
    setLevel(0)
    try {
      ExpoSpeechRecognitionModule.abort()
    } catch {
      // Nothing was listening; the local reset above is enough.
    }
  }, [clearSafetyTimer])

  useEffect(() => {
    return () => {
      clearSafetyTimer()
      try {
        ExpoSpeechRecognitionModule.abort()
      } catch {
        // unmounted before the native module was ever used
      }
    }
  }, [clearSafetyTimer])

  return {
    phase,
    interim,
    level,
    error: speechError,
    availability,
    start,
    stop,
    abort,
    isListening: phase === 'listening' || phase === 'requesting',
    isBusy: phase === 'stopping',
    dismissError: useCallback(() => setSpeechError(null), []),
  }
}
