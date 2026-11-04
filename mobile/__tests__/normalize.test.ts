import { USB_REVERSE_URL, describeHostIssue, normalizeBaseUrl, suggestFromDevServerHost } from '../src/api/normalize'

describe('server address', () => {
  it('accepts a tunnel address with no port', () => {
    const parsed = normalizeBaseUrl('https://unbraided-parachute-synergy.ngrok-free.dev')
    expect(parsed.ok).toBe(true)
    if (parsed.ok) expect(parsed.url).toBe('https://unbraided-parachute-synergy.ngrok-free.dev')
  })

  it('names the missing port for a bare LAN host', () => {
    const parsed = normalizeBaseUrl('http://192.168.1.20')
    expect(parsed.ok).toBe(false)
    if (!parsed.ok) expect(parsed.reason).toContain(':8080')
  })

  it('adds the scheme and keeps a port', () => {
    const parsed = normalizeBaseUrl(' 10.151.207.43:8080 ')
    expect(parsed.ok).toBe(true)
    if (parsed.ok) expect(parsed.url).toBe('http://10.151.207.43:8080')
  })

  it('rejects a port that is not a number', () => {
    expect(normalizeBaseUrl('http://192.168.1.20:abc').ok).toBe(false)
    expect(normalizeBaseUrl('http://192.168.1.20:99999').ok).toBe(false)
  })

  it('warns that a loopback address only works through the cable', () => {
    expect(describeHostIssue('http://127.0.0.1:8080')).toContain('adb reverse')
    expect(describeHostIssue('https://example.ngrok-free.dev')).toBeNull()
    expect(USB_REVERSE_URL).toBe('http://127.0.0.1:8080')
  })

  it('suggests the dev server host on port 8080', () => {
    expect(suggestFromDevServerHost('10.151.207.43:8081')).toBe('http://10.151.207.43:8080')
    expect(suggestFromDevServerHost(null)).toBeNull()
  })
})
