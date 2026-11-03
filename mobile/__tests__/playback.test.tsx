import { act, renderHook } from '@testing-library/react-native'
import { useReplyPlayer } from '../src/audio/playback'

const mockClips: string[] = []
const mockAudio = { didJustFinish: false, playing: false, error: null as string | null }

jest.mock('expo-audio', () => ({
  setAudioModeAsync: jest.fn(() => Promise.resolve()),
  useAudioPlayer: () => ({
    replace: (source: { uri: string }) => {
      mockClips.push(source.uri)
      mockAudio.didJustFinish = false
      mockAudio.error = null
    },
    play: () => {
      mockAudio.playing = true
    },
    pause: () => {
      mockAudio.playing = false
    },
    muted: false,
  }),
  useAudioPlayerStatus: () => mockAudio,
}))

const BASE = 'http://10.151.207.43:8080'

type Tick = { tick: number }

async function setup() {
  return renderHook((_props: Tick) => useReplyPlayer(BASE, false), { initialProps: { tick: 0 } })
}

/** expo-audio pushes playback status through a hook, so a finished clip has to be delivered with a render. */
async function finishClip(rerender: (props: Tick) => Promise<void>, tick: number) {
  await act(async () => {
    mockAudio.didJustFinish = true
    mockAudio.playing = false
    await rerender({ tick })
  })
}

/** A clip that cannot be opened arrives the same way: as a status pushed through a render. */
async function failClip(rerender: (props: Tick) => Promise<void>, tick: number, message: string) {
  await act(async () => {
    mockAudio.error = message
    await rerender({ tick })
  })
}

beforeEach(() => {
  mockClips.length = 0
  mockAudio.didJustFinish = false
  mockAudio.playing = false
  mockAudio.error = null
})

describe('reply player queue', () => {
  it('starts the first clip without waiting for the rest of the answer', async () => {
    const { result, rerender } = await setup()
    await act(async () => result.current.playSegments(['/audio/head.mp3'], 'a1'))

    expect(mockClips).toEqual([`${BASE}/audio/head.mp3`])
    expect(result.current.activeId).toBe('a1')
    expect(result.current.isPlaying).toBe(true)

    await finishClip(rerender, 1)
    expect(result.current.activeId).toBeNull()

    await act(async () => result.current.enqueue(['/audio/tail.mp3'], 'a1'))
    expect(mockClips).toEqual([`${BASE}/audio/head.mp3`, `${BASE}/audio/tail.mp3`])
    expect(result.current.activeId).toBe('a1')
  })

  it('advances once per finished clip, even across status re-renders', async () => {
    const { result, rerender } = await setup()
    await act(async () => result.current.playSegments(['/audio/a.mp3', '/audio/b.mp3', '/audio/c.mp3'], 'm1'))
    expect(mockClips).toEqual([`${BASE}/audio/a.mp3`])

    await finishClip(rerender, 1)
    await act(async () => rerender({ tick: 2 }))
    expect(mockClips).toEqual([`${BASE}/audio/a.mp3`, `${BASE}/audio/b.mp3`])
    expect(result.current.segment).toBe(1)

    await finishClip(rerender, 3)
    expect(mockClips).toHaveLength(3)
    expect(mockClips[2]).toBe(`${BASE}/audio/c.mp3`)

    await finishClip(rerender, 4)
    expect(mockClips).toHaveLength(3)
    expect(result.current.activeId).toBeNull()
  })

  it('never resumes a session the user stopped', async () => {
    const { result } = await setup()
    await act(async () => result.current.playSegments(['/audio/head.mp3'], 'a1'))
    await act(async () => result.current.stop())
    await act(async () => result.current.enqueue(['/audio/tail.mp3'], 'a1'))

    expect(mockClips).toEqual([`${BASE}/audio/head.mp3`])
    expect(result.current.activeId).toBeNull()
  })

  it('ignores late clips that belong to an earlier answer', async () => {
    const { result } = await setup()
    await act(async () => result.current.playSegments(['/audio/next.mp3'], 'a2'))
    await act(async () => result.current.enqueue(['/audio/stale.mp3'], 'a1'))

    expect(mockClips).toEqual([`${BASE}/audio/next.mp3`])
  })

  it('reports a missing clip instead of going silent', async () => {
    const { result } = await setup()
    await act(async () => result.current.playSegments([], 'a1'))
    expect(result.current.error).toBe('The server returned this answer without audio.')

    await act(async () => result.current.playSegments(null, 'a1'))
    expect(result.current.error).toBe('The server returned this answer without audio.')
  })

  it('skips one dead clip and plays the rest of the answer', async () => {
    const { result, rerender } = await setup()
    await act(async () => result.current.playSegments(['/audio/a.mp3', '/audio/b.mp3', '/audio/c.mp3'], 'm9'))
    expect(mockClips).toEqual([`${BASE}/audio/a.mp3`])

    // An evicted file or a download dropped mid-clip must cost one sentence, not
    // the whole reply, and one error must not skip the whole queue.
    await failClip(rerender, 1, 'file is no longer on the server')
    expect(mockClips).toEqual([`${BASE}/audio/a.mp3`, `${BASE}/audio/b.mp3`])
    expect(result.current.activeId).toBe('m9')
    expect(result.current.error).toBeNull()

    await finishClip(rerender, 2)
    expect(mockClips).toHaveLength(3)
    expect(mockClips[2]).toBe(`${BASE}/audio/c.mp3`)

    await finishClip(rerender, 3)
    expect(result.current.activeId).toBeNull()
    expect(result.current.error).toBe(
      '1 of 3 clips could not be played. The rest of that answer is on screen.'
    )
  })

  it('reports a failure once there is no session left to rescue', async () => {
    const { result, rerender } = await setup()
    await failClip(rerender, 1, 'the decoder refused the stream')

    expect(result.current.error).toBe('Audio could not be played: the decoder refused the stream')
    expect(result.current.activeId).toBeNull()
  })
})
