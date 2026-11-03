import type { AssistantStage, ChatMessage } from '../types'

export type ConversationState = { messages: ChatMessage[]; pendingId: string | null }

export type ConversationAction =
  | { type: 'hydrate'; messages: ChatMessage[] }
  | { type: 'system'; message: ChatMessage }
  | { type: 'ask'; question: ChatMessage; placeholder: ChatMessage }
  | { type: 'stage'; id: string; stage: AssistantStage }
  | { type: 'answer'; id: string; patch: Partial<ChatMessage> }
  | { type: 'clips'; id: string; audioUrls: string[] }
  | { type: 'voice'; id: string; patch: Partial<ChatMessage> }
  | { type: 'fail'; id: string; text: string }
  | { type: 'retry'; id: string }
  | { type: 'clear' }

export const EMPTY_CONVERSATION: ConversationState = { messages: [], pendingId: null }

const PERSIST_LIMIT = 40

export function conversationReducer(
  state: ConversationState,
  action: ConversationAction
): ConversationState {
  switch (action.type) {
    case 'hydrate':
      return { messages: action.messages, pendingId: null }

    case 'system':
      return { ...state, messages: [...state.messages, action.message] }

    case 'ask':
      return {
        messages: [...state.messages, action.question, action.placeholder],
        pendingId: action.placeholder.id,
      }

    case 'stage':
      return {
        ...state,
        messages: state.messages.map((message) =>
          message.id === action.id ? { ...message, stage: action.stage } : message
        ),
      }

    case 'answer':
      return {
        pendingId: null,
        messages: state.messages.map((message) =>
          message.id === action.id
            ? { ...message, ...action.patch, stage: 'voice' as AssistantStage, failed: false }
            : message
        ),
      }

    case 'clips':
      return {
        ...state,
        messages: state.messages.map((message) =>
          message.id === action.id ? { ...message, audioUrls: action.audioUrls } : message
        ),
      }

    case 'voice':
      return {
        pendingId: null,
        messages: state.messages.map((message) =>
          message.id === action.id
            ? { ...message, ...action.patch, stage: 'idle' as AssistantStage }
            : message
        ),
      }

    case 'fail':
      return {
        pendingId: null,
        messages: state.messages.map((message) =>
          message.id === action.id
            ? { ...message, text: action.text, stage: 'idle' as AssistantStage, failed: true }
            : message
        ),
      }

    case 'retry': {
      const index = state.messages.findIndex((message) => message.id === action.id)
      if (index < 1) return state
      const failed = state.messages[index]
      const placeholder = {
        ...failed,
        text: '',
        audioUrls: null,
        failed: false,
        stage: 'working' as AssistantStage,
      }
      return {
        messages: [...state.messages.slice(0, index), placeholder],
        pendingId: placeholder.id,
      }
    }

    case 'clear':
      return EMPTY_CONVERSATION

    default:
      return state
  }
}

export function persistableMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages
    .filter((message) => message.stage !== 'working')
    .slice(-PERSIST_LIMIT)
    .map((message) => ({
      id: message.id,
      role: message.role,
      text: message.text,
      createdAt: message.createdAt,
      sources: message.sources,
      outOfContext: message.outOfContext,
      noDocument: message.noDocument,
      audioUrls: message.audioUrls ?? null,
      voice: message.voice,
      latencyMs: message.latencyMs,
      timings: message.timings,
      engine: message.engine,
      score: message.score,
      failed: message.failed,
    }))
}

let counter = 0
export function makeId(prefix = 'm'): string {
  counter += 1
  return `${prefix}_${Date.now().toString(36)}_${counter.toString(36)}`
}
