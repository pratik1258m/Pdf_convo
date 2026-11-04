import { useEffect, useState } from 'react'
import { Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native'
import { ExpoSpeechRecognitionModule } from 'expo-speech-recognition'
import { colors, font, radius, spacing } from '../theme'
import { Btn, Chip, SectionTitle, Sheet, StatusDot } from './ui'
import { shortHost } from '../utils/format'
import type { ListenMode } from '../utils/format'
import type { HealthSnapshot, VoiceOption } from '../types'
import type { SpeechAvailability } from '../audio/speech'

export function VoiceSheet({
  visible,
  voices,
  selected,
  loading,
  serverDefault,
  listen,
  micLanguage,
  onClose,
  onSelect,
  onListen,
}: {
  visible: boolean
  voices: VoiceOption[]
  selected: string
  loading: boolean
  serverDefault?: string
  listen: ListenMode
  micLanguage: string
  onClose: () => void
  onSelect: (voice: VoiceOption) => void
  onListen: (mode: ListenMode) => void
}) {
  return (
    <Sheet
      visible={visible}
      onClose={onClose}
      title="Reply voice"
      subtitle="Answers are spoken by the server. A voice marked 'on the server' is made locally and starts speaking sooner than a network one."
    >
      <SectionTitle>Microphone language</SectionTitle>
      <View style={styles.listenRow}>
        <Chip label="Follow the voice" tone={listen === 'auto' ? 'accent' : 'default'} onPress={() => onListen('auto')} />
        <Chip label="English" tone={listen === 'en-IN' ? 'accent' : 'default'} onPress={() => onListen('en-IN')} />
        <Chip label="हिन्दी" tone={listen === 'hi-IN' ? 'accent' : 'default'} onPress={() => onListen('hi-IN')} />
      </View>
      <Text style={styles.muted}>
        {`Listening in ${micLanguage}. ` +
          (listen === 'auto'
            ? 'Supertonic speaks Hindi and English from one voice, so set this yourself if you plan to answer in Hindi.'
            : 'This overrides the voice, so you can speak one language and hear the other.')}
      </Text>

      <SectionTitle>Voices on this server</SectionTitle>
      {loading ? <Text style={styles.muted}>Loading voices from the server</Text> : null}
      {!loading && voices.length === 0 ? (
        <Text style={styles.muted}>The server did not return any voices.</Text>
      ) : null}
      {voices.map((voice) => {
        const active = voice.id === selected
        const usable = voice.available !== false
        const meta = [
          voice.locale,
          voice.gender,
          voice.local ? 'on the server' : 'over the network',
          serverDefault === voice.id ? 'server default' : '',
          voice.recommended && serverDefault !== voice.id ? 'recommended' : '',
        ]
          .filter(Boolean)
          .join(' • ')
        return (
          <Pressable
            key={voice.id}
            onPress={() => onSelect(voice)}
            disabled={!usable}
            accessibilityRole="radio"
            accessibilityState={{ selected: active, disabled: !usable }}
            style={({ pressed }) => [
              styles.voiceRow,
              {
                borderColor: active ? colors.accent : colors.border,
                opacity: !usable ? 0.45 : pressed ? 0.85 : 1,
              },
            ]}
          >
            <View style={{ flex: 1 }}>
              <Text style={styles.voiceName}>{voice.name}</Text>
              <Text style={styles.voiceMeta}>{meta}</Text>
              {!usable ? (
              <Text style={styles.voiceMeta}>Not available on this server.</Text>
            ) : null}
            </View>
            <View style={[styles.radio, { borderColor: active ? colors.accent : colors.border }]}>
              {active ? <View style={styles.radioDot} /> : null}
            </View>
          </Pressable>
        )
      })}
    </Sheet>
  )
}

export function SettingsSheet({
  visible,
  onClose,
  baseUrl,
  health,
  healthTone,
  healthDetail,
  onRecheck,
  autoSpeak,
  muted,
  telemetry,
  onToggleAutoSpeak,
  onToggleMute,
  onToggleTelemetry,
  speech,
  onOpenVoices,
  voiceName,
  micLanguage,
  voiceNote,
  onClearConversation,
}: {
  visible: boolean
  onClose: () => void
  baseUrl: string
  health: HealthSnapshot | null
  healthTone: 'ok' | 'warn' | 'danger' | 'idle'
  healthDetail: string | null
  onRecheck: () => void
  autoSpeak: boolean
  muted: boolean
  telemetry: boolean
  onToggleAutoSpeak: (value: boolean) => void
  onToggleMute: (value: boolean) => void
  onToggleTelemetry: (value: boolean) => void
  speech: SpeechAvailability
  onOpenVoices: () => void
  voiceName: string
  /** Which language the microphone listens in: the listening setting, or the
   *  chosen voice when that setting is left on `auto`. */
  micLanguage: string
  /** Where the audio is made, so latency is explainable without reading a log. */
  voiceNote?: string
  onClearConversation: () => void
}) {
  const [confirmClear, setConfirmClear] = useState(false)
  const [diagnostics, setDiagnostics] = useState<string[]>([])

  useEffect(() => {
    if (!visible) {
      setConfirmClear(false)
      return
    }
    let alive = true
    async function collect() {
      const lines: string[] = []
      try {
        lines.push(`Recognizer available: ${ExpoSpeechRecognitionModule.isRecognitionAvailable() ? 'yes' : 'no'}`)
        lines.push(`On-device recognition: ${ExpoSpeechRecognitionModule.supportsOnDeviceRecognition() ? 'yes' : 'no'}`)
        lines.push(`Can record audio: ${ExpoSpeechRecognitionModule.supportsRecording() ? 'yes' : 'no'}`)
        const services = ExpoSpeechRecognitionModule.getSpeechRecognitionServices()
        if (services.length) lines.push(`Services: ${services.join(', ')}`)
        const { installedLocales } = await ExpoSpeechRecognitionModule.getSupportedLocales({})
        lines.push(
          installedLocales.length
            ? `Installed languages: ${installedLocales.slice(0, 8).join(', ')}${installedLocales.length > 8 ? ' …' : ''}`
            : 'Installed languages: none reported'
        )
        const permission = await ExpoSpeechRecognitionModule.getPermissionsAsync()
        lines.push(`Microphone + speech permission: ${permission.status}`)
      } catch (error) {
        lines.push(error instanceof Error ? error.message : 'Diagnostics unavailable in this build.')
      }
      if (alive) setDiagnostics(lines)
    }
    void collect()
    return () => {
      alive = false
    }
  }, [visible])

  return (
    <Sheet visible={visible} onClose={onClose} title="Settings">
      <SectionTitle>Server</SectionTitle>
      <View style={styles.card}>
        <View style={styles.line}>
          <StatusDot tone={healthTone} />
          <Text style={styles.lineText} numberOfLines={1}>
            {shortHost(baseUrl)}
          </Text>
        </View>
        {health ? (
          <Text style={styles.subtle}>
            {health.current_document} • {health.total_chunks} chunks indexed
          </Text>
        ) : null}
        {telemetry && health?.engine ? <Text style={styles.subtle}>Answer engine: {health.engine}</Text> : null}
        {telemetry && health?.chunking?.chunks ? (
          <Text style={styles.subtle}>
            Chunking: median {health.chunking.words_p50} words, range {health.chunking.words_min}-
            {health.chunking.words_max}, {health.chunking.headings_tracked} headings carried,{' '}
            {health.chunking.short_runs_merged} short runs merged, {health.chunking.text_preserved_pct}% of the page
            text indexed.
          </Text>
        ) : null}
        {healthDetail ? <Text style={styles.warn}>{healthDetail}</Text> : null}
        <Btn label="Re-check connection" variant="soft" onPress={onRecheck} style={{ marginTop: spacing.md }} />
      </View>

      <SectionTitle>Playback</SectionTitle>
      <View style={styles.switchRow}>
        <Text style={styles.switchLabel}>Speak answers automatically</Text>
        <Switch value={autoSpeak} onValueChange={onToggleAutoSpeak} trackColor={{ true: colors.accent }} />
      </View>
      <View style={styles.switchRow}>
        <Text style={styles.switchLabel}>Mute all audio</Text>
        <Switch value={muted} onValueChange={onToggleMute} trackColor={{ true: colors.accent }} />
      </View>

      <SectionTitle>Developer</SectionTitle>
      <View style={styles.switchRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.switchLabel}>Show engine and timings</Text>
          <Text style={styles.subtle}>
            Puts stage times, match strength and the model or voice behind each reply on screen. Meant for testing the
            server, not for listening.
          </Text>
        </View>
        <Switch value={telemetry} onValueChange={onToggleTelemetry} trackColor={{ true: colors.accent }} />
      </View>

      <SectionTitle>Voice</SectionTitle>
      <Pressable onPress={onOpenVoices} style={styles.switchRow} accessibilityRole="button">
        <View style={{ flex: 1 }}>
          <Text style={styles.switchLabel}>Reply voice</Text>
          <Text style={styles.subtle}>
            {voiceName} • mic language {micLanguage}
            {voiceNote ? ` • ${voiceNote}` : ''}
          </Text>
        </View>
        <Text style={styles.chevron}>Change</Text>
      </Pressable>
      {speech.available ? null : (
        <Text style={styles.warn}>
          On-device voice input is not available in this build{speech.detail ? `: ${speech.detail}` : ''}. Use the
          microphone key on the on-screen keyboard, or install the development build.
        </Text>
      )}

      <SectionTitle>Recognition diagnostics</SectionTitle>
      {diagnostics.length === 0 ? <Text style={styles.subtle}>Reading device capabilities</Text> : null}
      {diagnostics.map((line) => (
        <Text key={line} style={styles.diagnostic}>
          {line}
        </Text>
      ))}

      <SectionTitle>Conversation</SectionTitle>
      <View style={styles.tagWrap}>
        <Chip label="Answers cite your file" />
        <Chip label="Anything else is labelled" tone="ok" />
      </View>
      <Btn
        label={confirmClear ? 'Tap again to erase history' : 'Clear conversation'}
        variant="danger"
        onPress={() => {
          if (confirmClear) {
            onClearConversation()
            setConfirmClear(false)
            onClose()
          } else {
            setConfirmClear(true)
          }
        }}
        style={{ marginTop: spacing.md }}
      />
    </Sheet>
  )
}

const styles = StyleSheet.create({
  listenRow: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: spacing.xs },
  voiceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.lg,
    marginBottom: spacing.sm + 2,
    backgroundColor: colors.surfaceAlt,
    gap: spacing.md,
  },
  voiceName: { color: colors.text, fontSize: font.md, fontWeight: '600' },
  voiceMeta: { color: colors.textMuted, fontSize: font.xs, marginTop: 2 },
  radio: { width: 20, height: 20, borderRadius: 10, borderWidth: 2, alignItems: 'center', justifyContent: 'center' },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: colors.accent },
  card: { backgroundColor: colors.surfaceAlt, borderRadius: radius.md, padding: spacing.lg, borderWidth: 1, borderColor: colors.border },
  line: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  lineText: { color: colors.text, fontSize: font.md, flex: 1 },
  subtle: { color: colors.textMuted, fontSize: font.sm, marginTop: spacing.xs + 2, lineHeight: 19 },
  warn: { color: colors.warn, fontSize: font.sm, marginTop: spacing.sm, lineHeight: 19 },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    gap: spacing.md,
  },
  switchLabel: { color: colors.text, fontSize: font.md },
  chevron: { color: colors.accent, fontSize: font.sm, fontWeight: '700' },
  diagnostic: { color: colors.textMuted, fontSize: font.xs, lineHeight: 18, marginBottom: 2 },
  tagWrap: { flexDirection: 'row', flexWrap: 'wrap' },
  muted: { color: colors.textMuted, fontSize: font.md },
})
