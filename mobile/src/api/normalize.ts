export type NormalizedUrl =
  | { ok: true; url: string; host: string }
  | { ok: false; reason: string }

const URL_PATTERN = /^(https?):\/\/([^\/?#\s]+)([\/?#].*)?$/i

export function normalizeBaseUrl(raw: string): NormalizedUrl {
  const trimmed = (raw || '').trim()
  if (!trimmed) return { ok: false, reason: 'Enter the server address shown when you run python3 server.py' }

  const withScheme = /^[a-z]+:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`
  if (!/^https?:\/\//i.test(withScheme)) {
    return { ok: false, reason: 'Only http and https addresses are supported' }
  }

  const match = URL_PATTERN.exec(withScheme)
  if (!match) return { ok: false, reason: 'That does not look like a valid address' }

  const scheme = match[1].toLowerCase()
  const host = match[2].toLowerCase()
  if (!host.includes(':')) return { ok: false, reason: 'Add the port, for example :8080' }
  const port = Number(host.slice(host.indexOf(':') + 1))
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    return { ok: false, reason: `"${host.split(':')[1]}" is not a valid port number` }
  }

  return { ok: true, url: `${scheme}://${host}`, host }
}

export function resolveAudioUrl(baseUrl: string, path?: string | null): string | null {
  if (!path) return null
  if (/^https?:\/\//i.test(path)) return path
  return `${baseUrl}${path.startsWith('/') ? '' : '/'}${path}`
}

export function describeHostIssue(baseUrl: string): string | null {
  const host = baseUrl.replace(/^https?:\/\//i, '').split(':')[0]
  if (host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === '[::1]') {
    return 'localhost points at the phone, not your Mac. Use the computer\'s LAN address, e.g. http://10.151.207.43:8080. On an Android emulator use http://10.0.2.2:8080.'
  }
  return null
}

export function suggestFromDevServerHost(hostUri?: string | null): string | null {
  if (!hostUri) return null
  const host = hostUri.replace(/^https?:\/\//i, '').split(':')[0]
  if (!host || host === 'localhost' || host === '127.0.0.1') return null
  return `http://${host}:8080`
}
