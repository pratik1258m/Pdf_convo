import { act, render, screen } from '@testing-library/react-native'
import { MessageBubble } from '../src/components/MessageBubble'
import type { ChatMessage } from '../src/types'

const base: ChatMessage = { id: 'a1', role: 'assistant', text: 'The file is students.xlsx', createdAt: 1 }

async function renderBubble(patch: Partial<ChatMessage>) {
  return render(
    <MessageBubble
      message={{ ...base, ...patch }}
      isPlaying={false}
      playingSegment={0}
      onReplay={() => undefined}
      onRetry={() => undefined}
    />
  )
}

describe('answer bubble', () => {
  it('shows the answer, its provenance and the measured stages', async () => {
    await renderBubble({
      sources: [8],
      score: 0.77,
      engine: 'extractive bm25 (no model configured)',
      latencyMs: 145,
      timings: { retrieval_ms: 0.3, answer_ms: 0.4, tts_ms: 900 },
      audioUrls: ['/audio/a.mp3', '/audio/b.mp3'],
    })

    expect(screen.getByText('The file is students.xlsx')).toBeTruthy()
    expect(screen.getByText('Page 8')).toBeTruthy()
    expect(screen.getByText(/match 0.77/)).toBeTruthy()
    expect(screen.getByText(/extractive bm25/)).toBeTruthy()
    expect(screen.getByText(/on screen 145 ms/)).toBeTruthy()
    expect(screen.getByText(/retrieval 0.3 ms/)).toBeTruthy()
    expect(screen.getByText(/speech 900 ms/)).toBeTruthy()
    expect(screen.getByText('Replay • 2 clips')).toBeTruthy()
  })

  it('hides timings the server never measured', async () => {
    await renderBubble({ text: 'Wait for alert presence.', sources: [3] })
    expect(screen.queryByText(/retrieval/)).toBeNull()
    expect(screen.queryByText(/speech/)).toBeNull()
    expect(screen.getByText('Page 3')).toBeTruthy()
  })

  it('labels an out-of-context answer instead of dressing it up as grounded', async () => {
    await renderBubble({ text: 'I could not locate this inside the manual.', outOfContext: true, sources: [] })
    expect(screen.getByText('Outside this document')).toBeTruthy()
  })

  it('says no document is loaded instead of claiming the topic is missing', async () => {
    const { unmount } = await renderBubble({
      text: 'Upload a PDF first, and I can answer from it.',
      noDocument: true,
    })
    expect(screen.getByText('No document loaded')).toBeTruthy()
    expect(screen.queryByText('Outside this document')).toBeNull()

    await act(async () => unmount())
  })

  it('keeps saying voice is pending while clips are still arriving', async () => {
    const { unmount } = await renderBubble({ stage: 'voice', audioUrls: null })
    expect(screen.getByText(/synthesizing the voice reply/)).toBeTruthy()

    await act(async () => unmount())
    await renderBubble({ stage: 'voice', audioUrls: ['/audio/a.mp3'] })
    expect(screen.getByText(/1 clip ready and more still synthesizing/)).toBeTruthy()
  })

  it('offers a retry for a failed turn', async () => {
    await renderBubble({ text: 'Could not reach the server', failed: true })
    expect(screen.getByText('Failed')).toBeTruthy()
    expect(screen.getByText('Retry')).toBeTruthy()
  })

  it('renders a system turn as a centred note', async () => {
    await renderBubble({ role: 'system', text: 'manual.pdf indexed in 83 ms: 15 pages, 48 chunks' })
    expect(screen.getByText(/indexed in 83 ms/)).toBeTruthy()
  })
})
