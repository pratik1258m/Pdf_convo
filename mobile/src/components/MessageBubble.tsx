import { memo } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { colors, font, radius, spacing } from '../theme'
import { Chip } from './ui'
import { formatLatency, timeLabel } from '../utils/format'
import type { ChatMessage } from '../types'

type Props = {
  message: ChatMessage
  isPlaying: boolean
  /** 1-based clip counter while this message plays; 0 when it is not the active one. */
  playingSegment: number
  onReplay: (message: ChatMessage) => void
  onRetry: (message: ChatMessage) => void
}

function MessageBubbleInner({ message, isPlaying, playingSegment, onReplay, onRetry }: Props) {
  if (message.role === 'system') {
    return (
      <View style={styles.systemWrap}>
        <Text style={[styles.systemText, message.failed ? { color: colors.danger } : null]}>{message.text}</Text>
        <Text style={styles.systemTime}>{timeLabel(message.createdAt)}</Text>
      </View>
    )
  }

  const isUser = message.role === 'user'
  const pending = message.stage === 'working'
  const makingVoice = message.stage === 'voice'
  const clips = message.audioUrls ?? []

  return (
    <View style={[styles.row, isUser ? styles.rowRight : styles.rowLeft]}>
      <View
        style={[
          styles.bubble,
          { backgroundColor: isUser ? colors.user : colors.assistant },
          message.failed ? { borderColor: colors.danger } : null,
        ]}
      >
        {pending ? (
          <Text style={styles.pendingText}>Searching the document for the exact passage…</Text>
        ) : (
          <Text style={[styles.text, isUser ? styles.userText : null]}>{message.text}</Text>
        )}

        {!isUser && !pending ? (
          <>
            <View style={styles.tagRow}>
              {message.failed ? <Chip label="Failed" tone="warn" /> : null}
              {message.noDocument && !message.failed ? <Chip label="No document loaded" tone="warn" /> : null}
              {message.outOfContext && !message.failed && !message.noDocument ? (
                <Chip label="Outside this document" tone="warn" />
              ) : null}
              {(message.sources ?? []).map((page) => (
                <Chip key={page} label={`Page ${page}`} tone={message.outOfContext ? 'default' : 'accent'} />
              ))}
            </View>

            {makingVoice ? (
              <Text style={styles.voicePending}>
                {clips.length
                  ? `Answer is on screen — speaking now, ${clips.length} clip${
                      clips.length === 1 ? '' : 's'
                    } ready and more still synthesizing…`
                  : 'Answer is on screen — synthesizing the voice reply…'}
              </Text>
            ) : null}

            <View style={styles.metaRow}>
              <Text style={styles.meta}>{timeLabel(message.createdAt)}</Text>
              {formatLatency(message.latencyMs) ? (
                <Text style={styles.meta}> • on screen {formatLatency(message.latencyMs)}</Text>
              ) : null}
              {formatLatency(message.timings?.retrieval_ms) ? (
                <Text style={styles.meta}> • retrieval {formatLatency(message.timings?.retrieval_ms)}</Text>
              ) : null}
              {formatLatency(message.timings?.tts_ms) ? (
                <Text style={styles.meta}> • speech {formatLatency(message.timings?.tts_ms)}</Text>
              ) : null}
              {typeof message.score === 'number' ? (
                <Text style={styles.meta}> • match {message.score.toFixed(2)}</Text>
              ) : null}
              {message.engine ? <Text style={styles.engine}> • {message.engine}</Text> : null}
              {message.voice ? <Text style={styles.meta}> • {message.voice.split('-').slice(0, 2).join('-')}</Text> : null}
              {clips.length && !message.failed ? (
                <Pressable
                  onPress={() => onReplay(message)}
                  hitSlop={8}
                  accessibilityRole="button"
                  accessibilityLabel={isPlaying ? 'Stop audio' : 'Play answer again'}
                  style={({ pressed }) => [styles.replay, { opacity: pressed ? 0.7 : 1 }]}
                >
                  <Text style={styles.replayText}>
                    {isPlaying
                      ? clips.length > 1
                        ? `Stop • clip ${playingSegment}/${clips.length}`
                        : 'Stop'
                      : clips.length > 1
                        ? `Replay • ${clips.length} clips`
                        : 'Replay'}
                  </Text>
                </Pressable>
              ) : null}
              {message.failed ? (
                <Pressable
                  onPress={() => onRetry(message)}
                  hitSlop={8}
                  accessibilityRole="button"
                  accessibilityLabel="Retry this question"
                  style={({ pressed }) => [styles.replay, { opacity: pressed ? 0.7 : 1 }]}
                >
                  <Text style={[styles.replayText, { color: colors.danger }]}>Retry</Text>
                </Pressable>
              ) : null}
            </View>
          </>
        ) : null}
      </View>
    </View>
  )
}

export const MessageBubble = memo(MessageBubbleInner)

const styles = StyleSheet.create({
  row: { flexDirection: 'row', marginBottom: spacing.md, paddingHorizontal: spacing.md },
  rowLeft: { justifyContent: 'flex-start' },
  rowRight: { justifyContent: 'flex-end' },
  bubble: {
    maxWidth: '86%',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.md,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  text: { color: colors.text, fontSize: font.md, lineHeight: 22 },
  userText: { color: colors.text },
  pendingText: { color: colors.textMuted, fontSize: font.md, fontStyle: 'italic' },
  tagRow: { flexDirection: 'row', flexWrap: 'wrap', marginTop: spacing.md },
  voicePending: { color: colors.accent, fontSize: font.xs, marginTop: spacing.sm, fontStyle: 'italic' },
  metaRow: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', marginTop: spacing.xs + 2 },
  meta: { color: colors.textFaint, fontSize: font.xs },
  engine: { color: colors.ok, fontSize: font.xs, fontWeight: '600' },
  replay: { marginLeft: spacing.md },
  replayText: { color: colors.accent, fontSize: font.xs, fontWeight: '700' },
  systemWrap: { alignItems: 'center', marginVertical: spacing.md, paddingHorizontal: spacing.xl },
  systemText: { color: colors.textMuted, fontSize: font.sm, textAlign: 'center', lineHeight: 19 },
  systemTime: { color: colors.textFaint, fontSize: 10, marginTop: 2 },
})
