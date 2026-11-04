import { askText } from '../src/api/backend'
import { EMPTY_CONVERSATION, askHistory, conversationReducer } from '../src/state/conversation'
import type { AskTextResponse, ChatMessage } from '../src/types'

const BASE = 'http://10.0.2.2:8080'

function message(patch: Partial<ChatMessage>): ChatMessage {
  return { id: 'm1', role: 'assistant', text: '', createdAt: 1, ...patch }
}

/** A turn as it sits on screen once the answer has landed and been spoken. */
function answered(question: string, answer: string, n: number): ChatMessage[] {
  return [
    message({ id: `q${n}`, role: 'user', text: question, createdAt: n * 10 }),
    message({ id: `a${n}`, text: answer, stage: 'idle', sources: [1], createdAt: n * 10 + 5 }),
  ]
}

/** A turn with no text yet: the question is on screen, the answer is not. */
function inFlight(question: string, n: number): ChatMessage[] {
  return [
    message({ id: `q${n}`, role: 'user', text: question, createdAt: n * 10 }),
    message({ id: `a${n}`, stage: 'working', createdAt: n * 10 + 1 }),
  ]
}

function replyTo(query: string): AskTextResponse {
  return {
    status: 'success',
    query,
    answer: 'Page 3 covers the calibration step [Page 3].',
    spoken_text: 'Page 3 covers the calibration step.',
    spoken_segments: ['Page 3 covers the calibration step.'],
    is_out_of_context: false,
    sources: [3],
    retrieval_score: 4.1,
    confidence_score: 0.8,
    engine: 'groq llama',
    timings: { retrieval_ms: 0.4, answer_ms: 620, total_ms: 621 },
  }
}

type Sent = { url: string; method?: string; body: string; headers?: Record<string, string> }

function bodyOf(request: Sent): Record<string, unknown> {
  return JSON.parse(request.body) as Record<string, unknown>
}

describe('ask history', () => {
  it('pairs each completed answer with the question it answered', () => {
    const history = askHistory([
      ...answered('What is this document about?', 'It is a selenium lab manual [Page 1].', 1),
      ...answered('What is on page 2?', 'Page 2 lists the setup [Page 2].', 2),
    ])
    expect(history).toEqual([
      { query: 'What is this document about?', answer: 'It is a selenium lab manual [Page 1].' },
      { query: 'What is on page 2?', answer: 'Page 2 lists the setup [Page 2].' },
    ])
  })

  it('leaves out notices, failed turns and answers that never landed', () => {
    const history = askHistory([
      message({ id: 's1', role: 'system', text: 'Uploading manual.pdf and indexing it' }),
      ...answered('Which website loads first?', 'Wikipedia is loaded first [Page 2].', 1),
      message({ id: 'q2', role: 'user', text: 'And the second lab?', createdAt: 20 }),
      message({ id: 'a2', text: 'Could not reach the server.', failed: true, stage: 'idle', createdAt: 21 }),
      message({ id: 'q3', role: 'user', text: '   ', createdAt: 30 }),
      message({ id: 'a3', text: 'blank answer', stage: 'idle', createdAt: 31 }),
      message({ id: 'q4', role: 'user', text: 'What about page 5?', createdAt: 40 }),
      message({ id: 'a4', text: '', stage: 'idle', createdAt: 41 }),
    ])
    expect(history).toEqual([{ query: 'Which website loads first?', answer: 'Wikipedia is loaded first [Page 2].' }])
  })

  it('keeps only the last four exchanges, most recent last', () => {
    const messages = [1, 2, 3, 4, 5, 6].flatMap((n) => answered(`Question ${n}`, `Answer ${n}`, n))
    expect(askHistory(messages)).toEqual([
      { query: 'Question 3', answer: 'Answer 3' },
      { query: 'Question 4', answer: 'Answer 4' },
      { query: 'Question 5', answer: 'Answer 5' },
      { query: 'Question 6', answer: 'Answer 6' },
    ])
  })

  it('omits the turn being asked, because the server reads the last real question', () => {
    // Mirrors the screen: history is built from committed state, then the question
    // and its placeholder are dispatched, then the request goes out.
    let state = conversationReducer(EMPTY_CONVERSATION, {
      type: 'ask',
      question: message({ id: 'q1', role: 'user', text: 'What is this document about?' }),
      placeholder: message({ id: 'a1', stage: 'working' }),
    })
    state = conversationReducer(state, {
      type: 'answer',
      id: 'a1',
      patch: { text: 'It is a selenium lab manual [Page 1].', sources: [1] },
    })

    const beforeComplaint = askHistory(state.messages)
    state = conversationReducer(state, {
      type: 'ask',
      question: message({ id: 'q2', role: 'user', text: 'no this is not a correct answer' }),
      placeholder: message({ id: 'a2', stage: 'working' }),
    })

    expect(beforeComplaint).toEqual([
      { query: 'What is this document about?', answer: 'It is a selenium lab manual [Page 1].' },
    ])
    // The complaint is on screen now, and still nothing of it is sent as history.
    expect(askHistory(state.messages)).toEqual(beforeComplaint)
  })

  it('returns nothing for an empty conversation', () => {
    expect(askHistory([])).toEqual([])
    expect(askHistory(inFlight('What is this document about?', 1))).toEqual([])
  })
})

describe('ask_text request body', () => {
  const sent: Sent[] = []
  const realFetch = globalThis.fetch

  // Re-installed for every test: the stub has to outlive the first request, because
  // the pushback turn and the first-ever question are the cases being proven.
  beforeEach(() => {
    sent.length = 0
    globalThis.fetch = ((input: unknown, init?: unknown) => {
      const options = (init ?? {}) as { method?: string; body?: unknown; headers?: Record<string, string> }
      sent.push({
        url: String(input),
        method: options.method,
        body: String(options.body ?? ''),
        headers: options.headers,
      })
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: () => Promise.resolve(JSON.stringify(replyTo(String(options.body ?? '')))),
      } as unknown as Response)
    }) as unknown as typeof fetch
  })

  afterEach(() => {
    globalThis.fetch = realFetch
  })

  it('carries the trimmed history with a short follow-up', async () => {
    const messages = [
      ...answered('What is this document about?', 'It is a selenium lab manual [Page 1].', 1),
      ...answered('What is on page 2?', 'Page 2 lists the setup [Page 2].', 2),
    ]
    await askText(BASE, { query: 'and page 3?', history: askHistory(messages) })

    expect(sent).toHaveLength(1)
    expect(sent[0].url).toBe(`${BASE}/api/ask_text`)
    expect(sent[0].method).toBe('POST')
    expect(sent[0].headers?.['Content-Type']).toBe('application/json')
    expect(bodyOf(sent[0])).toEqual({
      query: 'and page 3?',
      history: [
        { query: 'What is this document about?', answer: 'It is a selenium lab manual [Page 1].' },
        { query: 'What is on page 2?', answer: 'Page 2 lists the setup [Page 2].' },
      ],
    })
  })

  it('sends the questioned turn when the user says the answer is wrong', async () => {
    const messages = [...answered('Summarise the first page', 'The first page is an introduction [Page 1].', 1)]
    await askText(BASE, { query: 'no this is not a correct answer', history: askHistory(messages) })

    expect(bodyOf(sent[0])).toEqual({
      query: 'no this is not a correct answer',
      history: [{ query: 'Summarise the first page', answer: 'The first page is an introduction [Page 1].' }],
    })
  })

  it('trims the list to the last four exchanges on the wire', async () => {
    const messages = [1, 2, 3, 4, 5, 6].flatMap((n) => answered(`Question ${n}`, `Answer ${n}`, n))
    await askText(BASE, { query: 'What are the key points?', history: askHistory(messages) })

    const history = bodyOf(sent[0]).history as { query: string }[]
    expect(history).toHaveLength(4)
    expect(history[0].query).toBe('Question 3')
    expect(history[3].query).toBe('Question 6')
  })

  it('leaves the field out when there is nothing completed to send', async () => {
    await askText(BASE, { query: 'What is this document about?', history: askHistory([]) })

    const body = bodyOf(sent[0])
    expect(body.query).toBe('What is this document about?')
    expect('history' in body).toBe(false)
  })
})
