import {
  EMPTY_CONVERSATION,
  conversationReducer,
  makeId,
  persistableMessages,
} from '../src/state/conversation'
import type { ChatMessage } from '../src/types'
import { formatLatency, localeForVoice, micLocale, plural, resolveMicLocale, timeLabel } from '../src/utils/format'

function message(patch: Partial<ChatMessage>): ChatMessage {
  return { id: 'm1', role: 'assistant', text: '', createdAt: 1, ...patch }
}

describe('conversation reducer', () => {
  it('reserves a placeholder while the question is in flight', () => {
    const next = conversationReducer(EMPTY_CONVERSATION, {
      type: 'ask',
      question: message({ id: 'q1', role: 'user', text: 'What is the file name?' }),
      placeholder: message({ id: 'a1', stage: 'working' }),
    })
    expect(next.pendingId).toBe('a1')
    expect(next.messages.map((entry) => entry.id)).toEqual(['q1', 'a1'])
  })

  it('keeps the text stage pending until every clip has landed', () => {
    let state = conversationReducer(EMPTY_CONVERSATION, {
      type: 'ask',
      question: message({ id: 'q1', role: 'user', text: 'Summarise page 4' }),
      placeholder: message({ id: 'a1', stage: 'working' }),
    })

    state = conversationReducer(state, {
      type: 'answer',
      id: 'a1',
      patch: { text: 'Module 1 - Practical 4 [Page 4]', timings: { retrieval_ms: 0.3 }, latencyMs: 145 },
    })
    expect(state.pendingId).toBeNull()
    expect(state.messages[1].stage).toBe('voice')
    expect(state.messages[1].text).toContain('Practical 4')

    state = conversationReducer(state, { type: 'clips', id: 'a1', audioUrls: ['/audio/head.mp3'] })
    expect(state.messages[1].stage).toBe('voice')
    expect(state.messages[1].audioUrls).toEqual(['/audio/head.mp3'])

    state = conversationReducer(state, {
      type: 'voice',
      id: 'a1',
      patch: { audioUrls: ['/audio/head.mp3', '/audio/tail.mp3'], timings: { retrieval_ms: 0.3, tts_ms: 900 } },
    })
    expect(state.messages[1].stage).toBe('idle')
    expect(state.messages[1].audioUrls).toHaveLength(2)
    expect(state.messages[1].timings).toEqual({ retrieval_ms: 0.3, tts_ms: 900 })
  })

  it('retries in place without duplicating the question', () => {
    const seeded = conversationReducer(EMPTY_CONVERSATION, {
      type: 'ask',
      question: message({ id: 'q1', role: 'user', text: 'Which website loads first?' }),
      placeholder: message({ id: 'a1', stage: 'working' }),
    })
    const failedTurn = conversationReducer(seeded, { type: 'fail', id: 'a1', text: 'Server unreachable' })
    expect(failedTurn.messages[1]).toMatchObject({ failed: true, text: 'Server unreachable' })

    const retried = conversationReducer(failedTurn, { type: 'retry', id: 'a1' })
    expect(retried.messages).toHaveLength(2)
    expect(retried.messages[1]).toMatchObject({ id: 'a1', text: '', stage: 'working', failed: false, audioUrls: null })
    expect(retried.pendingId).toBe('a1')

    const answered = conversationReducer(retried, {
      type: 'answer',
      id: 'a1',
      patch: { text: 'Wikipedia is loaded first [Page 2].', sources: [2] },
    })
    expect(answered.messages).toHaveLength(2)
    expect(answered.messages[0].role).toBe('user')
    expect(answered.messages[1].audioUrls).toBeNull()
  })

  it('drops unfinished turns from storage but keeps measured timings', () => {
    const stored = persistableMessages([
      message({ id: 'q1', role: 'user', text: 'hello', latencyMs: 12 }),
      message({
        id: 'a1',
        text: 'answer',
        stage: 'voice',
        audioUrls: ['/audio/a.mp3'],
        timings: { retrieval_ms: 0.4, tts_ms: 1200 },
        engine: 'extractive bm25',
      }),
      message({ id: 'a2', text: '', stage: 'working' }),
    ])
    expect(stored.map((entry) => entry.id)).toEqual(['q1', 'a1'])
    expect(stored[1].stage).toBeUndefined()
    expect(stored[1].audioUrls).toEqual(['/audio/a.mp3'])
    expect(stored[1].timings).toEqual({ retrieval_ms: 0.4, tts_ms: 1200 })
  })

  it('makes unique ids', () => {
    expect(makeId('a')).not.toBe(makeId('a'))
  })
})

describe('format helpers', () => {
  it('renders latency in the unit a human reads', () => {
    expect(formatLatency(0.3)).toBe('0.3 ms')
    expect(formatLatency(145.6)).toBe('146 ms')
    expect(formatLatency(2259.8)).toBe('2.3 s')
    expect(formatLatency(undefined)).toBeNull()
    expect(formatLatency(-1)).toBeNull()
  })

  it('derives the microphone locale from the voice id', () => {
    expect(localeForVoice('en-IN-NeerjaNeural')).toBe('en-IN')
    expect(localeForVoice('hi-IN-SwaraNeural')).toBe('hi-IN')
    expect(localeForVoice('broken')).toBe('en-IN')
    expect(localeForVoice('hi_IN-pratham-medium')).toBe('hi-IN')
    expect(localeForVoice('en_US-ryan-medium')).toBe('en-US')
    expect(localeForVoice('supertonic-M1')).toBe('en-IN')
  })

  it('trusts the server label over a shape guessed from a local voice id', () => {
    // A Piper or Supertonic id is not shaped like an Edge id, so parsing it
    // would set the microphone to the wrong language on a perfectly good voice.
    expect(micLocale({ locale: 'hi-IN' }, 'hi_IN-pratham-medium')).toBe('hi-IN')
    expect(micLocale({ locale: 'en-IN' }, 'supertonic-M1')).toBe('en-IN')
    expect(micLocale(undefined, 'hi-IN-SwaraNeural')).toBe('hi-IN')
    expect(micLocale(undefined, 'en_US-ryan-medium')).toBe('en-US')
  })

  it('lets an explicit listening language beat the voice', () => {
    // The server's one Hindi-and-English voice reports en-IN, so hearing Hindi
    // with it has to be a setting, not something derived from the reply voice.
    expect(resolveMicLocale('hi-IN', { locale: 'en-IN' }, 'supertonic-M1')).toBe('hi-IN')
    expect(resolveMicLocale('en-IN', { locale: 'hi-IN' }, 'hi_IN-pratham-medium')).toBe('en-IN')
    expect(resolveMicLocale('auto', { locale: 'en-IN' }, 'supertonic-M1')).toBe('en-IN')
    expect(resolveMicLocale('auto', undefined, 'hi-IN-SwaraNeural')).toBe('hi-IN')
  })

  it('pluralises and formats a clock label', () => {
    expect(plural(1, 'page', 'pages')).toBe('1 page')
    expect(plural(15, 'page', 'pages')).toBe('15 pages')
    expect(timeLabel(new Date(2026, 9, 31, 13, 5).getTime())).toBe('1:05 PM')
  })
})
