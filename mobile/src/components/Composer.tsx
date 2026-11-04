import { useEffect, useRef, type ComponentRef } from 'react'
import {
  Animated,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native'
import { colors, font, radius, spacing } from '../theme'
import type { SpeechPhase } from '../audio/speech'

const BAR_COUNT = 7

function LevelMeter({ level }: { level: number }) {
  const animations = useRef<Animated.Value[]>(
    Array.from({ length: BAR_COUNT }, () => new Animated.Value(0))
  ).current

  useEffect(() => {
    animations.forEach((value, index) => {
      // Spread one captured mic level across the bar row so the meter reads as a waveform.
      const offset = Math.abs(index - (BAR_COUNT - 1) / 2) / (BAR_COUNT / 2)
      const target = Math.max(0.08, Math.min(1, level * (1 - offset * 0.55)))
      Animated.timing(value, { toValue: target, duration: 90, useNativeDriver: false }).start()
    })
  }, [level, animations])

  return (
    <View style={styles.meterRow} accessibilityLabel={`Microphone level ${Math.round(level * 100)} percent`}>
      {animations.map((value, index) => (
        <Animated.View
          key={index}
          style={[
            styles.meterBar,
            { height: value.interpolate({ inputRange: [0, 1], outputRange: [6, 26] }) },
          ]}
        />
      ))}
    </View>
  )
}

type Props = {
  draft: string
  onDraftChange: (value: string) => void
  onSend: () => void
  onTalkStart: () => void
  onTalkStop: () => void
  onTalkCancel: () => void
  phase: SpeechPhase
  level: number
  listening: boolean
  busy: boolean
  voiceAvailable: boolean
  notice: string | null
  onDismissNotice: () => void
  autoFocusSignal: number
}

export function Composer({
  draft,
  onDraftChange,
  onSend,
  onTalkStart,
  onTalkStop,
  onTalkCancel,
  phase,
  level,
  listening,
  busy,
  voiceAvailable,
  notice,
  onDismissNotice,
  autoFocusSignal,
}: Props) {
  const inputRef = useRef<ComponentRef<typeof TextInput> | null>(null)
  const canSend = draft.trim().length > 0 && !busy && !listening

  useEffect(() => {
    if (autoFocusSignal > 0) inputRef.current?.focus()
  }, [autoFocusSignal])

  const statusText = listening
    ? 'Listening. Tap the mic again to send.'
    : phase === 'stopping'
      ? 'Finalising what I heard'
      : busy
        ? 'Thinking'
        : voiceAvailable
          ? 'Tap the mic to speak, tap again to send. Or type below.'
          : 'Voice input needs the built app. Tap the field and use the keyboard microphone.'

  return (
    <View style={styles.wrap}>
      {notice ? (
        <Pressable style={styles.notice} onPress={onDismissNotice} accessibilityRole="button">
          <Text style={styles.noticeText} numberOfLines={2}>
            {notice}
          </Text>
          <Text style={styles.noticeClose}>Dismiss</Text>
        </Pressable>
      ) : null}

      {listening ? (
        <View style={styles.listeningRow}>
          <LevelMeter level={level} />
          <Text style={styles.interim} numberOfLines={2}>
            {draft ? draft : 'Say your question'}
          </Text>
          <Pressable onPress={onTalkCancel} hitSlop={10} accessibilityRole="button" accessibilityLabel="Cancel voice input">
            <Text style={styles.cancel}>Cancel</Text>
          </Pressable>
        </View>
      ) : null}

      <View style={styles.row}>
        <Pressable
          onPress={listening ? onTalkStop : onTalkStart}
          accessibilityRole="button"
          accessibilityLabel={listening ? 'Stop listening and send the question' : 'Start speaking'}
          accessibilityState={{ disabled: !voiceAvailable }}
          style={({ pressed }) => [
            styles.mic,
            {
              backgroundColor: listening ? colors.accent : colors.surfaceAlt,
              borderColor: listening ? colors.accent : colors.border,
              opacity: pressed && !listening ? 0.85 : 1,
            },
          ]}
        >
          <View style={[styles.micCore, { backgroundColor: listening ? colors.accentInk : colors.accent }]} />
        </Pressable>

        <TextInput
          ref={inputRef}
          value={draft}
          onChangeText={onDraftChange}
          editable={!listening}
          placeholder="Ask about the document"
          placeholderTextColor={colors.textFaint}
          style={styles.input}
          returnKeyType="send"
          submitBehavior="submit"
          onSubmitEditing={onSend}
          accessibilityLabel="Question text"
          multiline={false}
        />

        <Pressable
          onPress={canSend ? onSend : undefined}
          accessibilityRole="button"
          accessibilityLabel="Send question"
          accessibilityState={{ disabled: !canSend }}
          style={({ pressed }) => [
            styles.send,
            { backgroundColor: canSend ? colors.accent : colors.surfaceHi, opacity: pressed && canSend ? 0.85 : 1 },
          ]}
        >
          <Text style={[styles.sendText, { color: canSend ? colors.accentInk : colors.textFaint }]}>Send</Text>
        </Pressable>
      </View>

      <Text style={styles.hint}>{statusText}</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  wrap: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.borderSoft,
  },
  notice: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(242, 169, 59, 0.12)',
    borderColor: colors.warn,
    borderWidth: 1,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginBottom: spacing.sm,
    gap: spacing.md,
  },
  noticeText: { color: colors.warn, fontSize: font.sm, flex: 1, lineHeight: 19 },
  noticeClose: { color: colors.textMuted, fontSize: font.xs, fontWeight: '700' },
  listeningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginBottom: spacing.sm,
    minHeight: 30,
  },
  meterRow: { flexDirection: 'row', alignItems: 'center', gap: 3, height: 28, width: 64 },
  meterBar: { width: 5, borderRadius: 3, backgroundColor: colors.accent },
  interim: { flex: 1, color: colors.text, fontSize: font.md },
  cancel: { color: colors.textMuted, fontSize: font.xs, fontWeight: '700' },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  mic: {
    width: 50,
    height: 50,
    borderRadius: radius.pill,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  micCore: { width: 16, height: 16, borderRadius: 8 },
  input: {
    flex: 1,
    minHeight: 50,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.bg,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    color: colors.text,
    fontSize: font.md,
  },
  send: {
    minWidth: 62,
    height: 50,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.md,
  },
  sendText: { fontSize: font.md, fontWeight: '700' },
  hint: { color: colors.textFaint, fontSize: font.xs, marginTop: spacing.sm, lineHeight: 17 },
})
