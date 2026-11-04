import { useMemo, useState } from 'react'
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native'
import Constants from 'expo-constants'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { BackendError, getHealth } from '../api/backend'
import { USB_REVERSE_URL, describeHostIssue, normalizeBaseUrl, suggestFromDevServerHost } from '../api/normalize'
import type { HealthSnapshot } from '../types'
import type { Settings } from '../state/settings'
import { colors, font, radius, spacing } from '../theme'
import { Btn, Chip, SectionTitle, StatusDot } from './ui'

export function ConnectScreen({
  settings,
  onConnect,
}: {
  settings: Settings
  onConnect: (url: string, health: HealthSnapshot) => void
}) {
  const insets = useSafeAreaInsets()
  const suggestion = useMemo(
    () => suggestFromDevServerHost(Constants.expoConfig?.hostUri ?? null),
    []
  )
  const [input, setInput] = useState(settings.baseUrl || suggestion || '')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; title: string; body: string } | null>(null)
  const [health, setHealth] = useState<HealthSnapshot | null>(null)

  const parsed = useMemo(() => normalizeBaseUrl(input.trim()), [input])
  const hostIssue = describeHostIssue(input.trim())

  async function connect() {
    if (!parsed.ok) {
      setResult({ ok: false, title: 'Check the address', body: parsed.reason })
      return
    }
    setBusy(true)
    setResult(null)
    try {
      const snapshot = await getHealth(parsed.url)
      setHealth(snapshot)
      setResult({
        ok: true,
        title: 'Server reachable',
        body: `${snapshot.current_document} indexed • ${snapshot.total_chunks} chunks • ${snapshot.available_voices} voices`,
      })
      onConnect(parsed.url, snapshot)
    } catch (error) {
      setHealth(null)
      setResult({
        ok: false,
        title: error instanceof BackendError ? `Could not connect (${error.kind})` : 'Could not connect',
        body: error instanceof Error ? error.message : 'Unknown error',
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        contentContainerStyle={[styles.content, { paddingTop: insets.top + spacing.xxl }]}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.brandRow}>
          <StatusDot tone={result?.ok ? 'ok' : result ? 'danger' : 'idle'} />
          <Text style={styles.brand}>Convo PDF</Text>
        </View>
        <Text style={styles.headline}>Talk to your PDF, out loud.</Text>
        <Text style={styles.sub}>
          Point the app at the voice server running on your computer. Upload a document, press the mic, and the
          answer comes back as speech.
        </Text>

        <Text style={styles.label}>Server address</Text>
        <TextInput
          value={input}
          onChangeText={setInput}
          autoCapitalize="none"
          autoCorrect={false}
          spellCheck={false}
          placeholder="http://192.168.1.20:8080"
          placeholderTextColor={colors.textFaint}
          style={styles.input}
          keyboardType="url"
          returnKeyType="go"
          onSubmitEditing={connect}
          accessibilityLabel="Server address"
        />

        {hostIssue ? <Text style={styles.warnText}>{hostIssue}</Text> : null}

        {suggestion && suggestion !== input ? (
          <View style={styles.suggestionRow}>
            <Text style={styles.suggestionText}>Dev server host detected</Text>
            <Chip label={suggestion} tone="accent" onPress={() => setInput(suggestion)} />
          </View>
        ) : null}

        {!suggestion && USB_REVERSE_URL !== input ? (
          <View style={styles.suggestionRow}>
            <Text style={styles.suggestionText}>USB cable with port forwarding</Text>
            <Chip label={USB_REVERSE_URL.replace(/^https?:\/\//i, '')} onPress={() => setInput(USB_REVERSE_URL)} />
          </View>
        ) : null}

        {settings.recentUrls.length > 0 ? (
          <View style={styles.recentRow}>
            {settings.recentUrls.map((url) => (
              <Chip key={url} label={url.replace(/^https?:\/\//i, '')} onPress={() => setInput(url)} />
            ))}
          </View>
        ) : null}

        <Btn label={busy ? 'Checking server' : 'Test and connect'} onPress={connect} loading={busy} disabled={busy} />
        {result ? (
          <View style={[styles.resultCard, { borderColor: result.ok ? colors.ok : colors.danger }]}>
            <Text style={[styles.resultTitle, { color: result.ok ? colors.ok : colors.danger }]}>
              {result.title}
            </Text>
            <Text style={styles.resultBody}>{result.body}</Text>
            {health ? (
              <View style={styles.healthGrid}>
                <Metric label="Document" value={health.current_document} />
                <Metric label="Chunks" value={String(health.total_chunks)} />
                <Metric label="Voices" value={String(health.available_voices)} />
              </View>
            ) : null}
          </View>
        ) : null}

        {/* Only for an address that is well formed but did not answer: opening the
            app with a malformed or localhost address would break every request. */}
        {result && !result.ok && parsed.ok && !hostIssue ? (
          <Btn label="Open the app anyway" variant="ghost" style={styles.anyway} onPress={() => onConnect(parsed.url, emptyHealth())} />
        ) : null}

        <SectionTitle>Getting the address</SectionTitle>
        <Facts
          lines={[
            '1. Start the server: python3 server.py',
            '2. On the same Wi-Fi: run ipconfig getifaddr en0 on the Mac, then enter http://<that address>:8080',
            '3. From anywhere: run ngrok http 8080 and enter the https:// address it prints',
            '4. Over the USB cable: run adb reverse tcp:8080 tcp:8080, then use the 127.0.0.1 chip above',
          ]}
        />
      </ScrollView>
    </KeyboardAvoidingView>
  )
}

function emptyHealth(): HealthSnapshot {
  return { status: 'unknown', current_document: 'None', total_chunks: 0, available_voices: 0 }
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricValue} numberOfLines={2}>
        {value}
      </Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
  )
}

function Facts({ lines }: { lines: string[] }) {
  return (
    <View style={styles.facts}>
      {lines.map((line) => (
        <Pressable key={line}>
          <Text style={styles.fact}>{line}</Text>
        </Pressable>
      ))}
    </View>
  )
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  content: { paddingHorizontal: spacing.xl, paddingBottom: spacing.xxl, maxWidth: 620, width: '100%', alignSelf: 'center' },
  brandRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  brand: { color: colors.textMuted, fontSize: font.sm, fontWeight: '700', letterSpacing: 1.6, textTransform: 'uppercase', marginLeft: spacing.sm },
  headline: { color: colors.text, fontSize: font.xxl, fontWeight: '800', marginTop: spacing.lg, lineHeight: 34 },
  sub: { color: colors.textMuted, fontSize: font.md, lineHeight: 22, marginTop: spacing.md },
  label: { color: colors.textFaint, fontSize: font.xs, fontWeight: '700', letterSpacing: 0.6, marginTop: spacing.xl, marginBottom: spacing.sm, textTransform: 'uppercase' },
  input: {
    minHeight: 50,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    paddingHorizontal: spacing.lg,
    color: colors.text,
    fontSize: font.md,
  },
  warnText: { color: colors.warn, fontSize: font.sm, marginTop: spacing.sm, lineHeight: 20 },
  suggestionRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: spacing.md, gap: spacing.sm },
  suggestionText: { color: colors.textFaint, fontSize: font.xs, flex: 1 },
  recentRow: { flexDirection: 'row', flexWrap: 'wrap', marginTop: spacing.md },
  resultCard: { marginTop: spacing.lg, borderWidth: 1, borderRadius: radius.md, padding: spacing.lg, backgroundColor: colors.surface },
  resultTitle: { fontSize: font.md, fontWeight: '700' },
  resultBody: { color: colors.textMuted, fontSize: font.sm, marginTop: spacing.xs + 2, lineHeight: 20 },
  healthGrid: { flexDirection: 'row', gap: spacing.md, marginTop: spacing.md, flexWrap: 'wrap' },
  metric: { flex: 1, minWidth: 90 },
  metricValue: { color: colors.text, fontSize: font.md, fontWeight: '700' },
  metricLabel: { color: colors.textFaint, fontSize: font.xs, marginTop: 2 },
  anyway: { marginTop: spacing.md },
  facts: { gap: spacing.sm + 2 },
  fact: { color: colors.textMuted, fontSize: font.sm, lineHeight: 20 },
})
