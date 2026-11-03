import { useCallback, useEffect, useState } from 'react'
import { normalizeBaseUrl } from '../api/normalize'
import { readJson, writeJson } from './storage'
import type { ListenMode } from '../utils/format'

export type Settings = {
  baseUrl: string
  voiceId: string
  listen: ListenMode
  autoSpeak: boolean
  muted: boolean
  recentUrls: string[]
}

const SETTINGS_KEY = 'convopdf.settings.v1'

export const DEFAULT_SETTINGS: Settings = {
  baseUrl: '',
  // Empty means "however the server speaks fastest": the phone follows
  // /api/health's default_voice until an actual voice is chosen here.
  voiceId: '',
  listen: 'auto',
  autoSpeak: true,
  muted: false,
  recentUrls: [],
}

/** A store written by an older build must never be the reason the app will not open. */
function sanitize(stored: Settings): Settings {
  // Only an absolute address with a port is ever handed to the chat screen, so a
  // half-typed or legacy value lands on the connect screen instead of failing fetch().
  const address = normalizeBaseUrl(stored.baseUrl ?? '')
  return {
    ...DEFAULT_SETTINGS,
    ...stored,
    baseUrl: address.ok ? address.url : '',
    voiceId: typeof stored.voiceId === 'string' ? stored.voiceId : DEFAULT_SETTINGS.voiceId,
    listen: stored.listen === 'en-IN' || stored.listen === 'hi-IN' ? stored.listen : 'auto',
    recentUrls: Array.isArray(stored.recentUrls)
      ? stored.recentUrls.filter((url): url is string => typeof url === 'string').slice(0, 4)
      : [],
  }
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let alive = true
    readJson<Settings>(SETTINGS_KEY, DEFAULT_SETTINGS).then((stored) => {
      if (!alive) return
      setSettings(sanitize(stored))
      setReady(true)
    })
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (ready) void writeJson(SETTINGS_KEY, settings)
  }, [settings, ready])

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  const rememberUrl = useCallback((url: string) => {
    setSettings((prev) => ({
      ...prev,
      baseUrl: url,
      recentUrls: [url, ...prev.recentUrls.filter((entry) => entry !== url)].slice(0, 4),
    }))
  }, [])

  return { settings, ready, update, rememberUrl }
}
