import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import {
  AppState,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import * as DocumentPicker from 'expo-document-picker'
import * as Haptics from 'expo-haptics'
import { BackendError, askText, getDocument, getHealth, getVoices, speakSegments, uploadPdf } from '../api/backend'
import { describeHostIssue } from '../api/normalize'
import { useReplyPlayer } from '../audio/playback'
import { useSpeechRecognition } from '../audio/speech'
import { EMPTY_CONVERSATION, askHistory, conversationReducer, makeId, persistableMessages } from '../state/conversation'
import { readArray, writeJson } from '../state/storage'
import { colors, font, radius, spacing } from '../theme'
import { formatLatency, resolveMicLocale, plural, shortHost } from '../utils/format'
import type { ChatMessage, DocumentStats, HealthSnapshot, SpeakResponse, Timings, VoiceOption } from '../types'
import type { Settings } from '../state/settings'
import { MessageBubble } from './MessageBubble'
import { Composer } from './Composer'
import { SettingsSheet, VoiceSheet } from './Sheets'
import { Btn, Chip, StatusDot } from './ui'

const HISTORY_KEY = 'convopdf.history.v1'

const STARTERS = [
  'What is this document about?',
  'Summarise the first page',
  'What are the key points?',
  'List the headings in this document',
]

function describeError(error: unknown): string {
  if (error instanceof BackendError) return error.message
  if (error instanceof Error) return error.message
  return 'Something went wrong while asking the server.'
}

/** A `partial` reply speaks fewer sentences than the answer has, and says so. */
function partialSpeechNotice(data: SpeakResponse): string | null {
  if (data.status !== 'partial') return null
  const missing = Math.max(1, (data.requested ?? 0) - data.segments.length)
  const reason = data.errors?.find(Boolean)
  return `Part of that answer could not be spoken (${plural(missing, 'sentence', 'sentences')} missing${
    reason ? `: ${reason}` : ''
  }). Everything is on screen.`
}

export function ChatScreen({
  baseUrl,
  settings,
  update,
  onChangeServer,
}: {
  baseUrl: string
  settings: Settings
  update: (patch: Partial<Settings>) => void
  onChangeServer: () => void
}) {
  const insets = useSafeAreaInsets()
  const [conversation, dispatch] = useReducer(conversationReducer, EMPTY_CONVERSATION)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<HealthSnapshot | null>(null)
  const [healthTone, setHealthTone] = useState<'ok' | 'warn' | 'danger' | 'idle'>('idle')
  const [healthDetail, setHealthDetail] = useState<string | null>(null)
  const [voices, setVoices] = useState<VoiceOption[]>([])
  const [notice, setNotice] = useState<string | null>(null)
  const [stats, setStats] = useState<DocumentStats | null>(null)
  const [uploading, setUploading] = useState(false)
  const [showVoices, setShowVoices] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [focusSignal, setFocusSignal] = useState(0)
  const [hydrated, setHydrated] = useState(false)

  const askRef = useRef<(question: string) => void>(() => {})
  // The transcript of the screen as of the last commit, read at ask time so the
  // question being dispatched right now is not in the history that goes with it.
  const messagesRef = useRef<ChatMessage[]>(conversation.messages)

  useEffect(() => {
    messagesRef.current = conversation.messages
  }, [conversation.messages])
  const player = useReplyPlayer(baseUrl, settings.muted)
  // No voice picked yet means the server decides, so the phone gets whichever
  // speaker this machine can make fastest instead of a hard-coded network voice.
  const voiceId = settings.voiceId || health?.default_voice || 'en-IN-PrabhatNeural'
  const selectedVoice = voices.find((voice) => voice.id === voiceId)
  const locale = resolveMicLocale(settings.listen, selectedVoice, voiceId)
  // The document the phone is actually talking about: the upload response first,
  // because health can still be reporting the file that was loaded before this one.
  const documentName = stats?.filename || (health && health.current_document !== 'None' ? health.current_document : '')

  const handleTranscript = useCallback((text: string, isFinal: boolean) => {
    if (isFinal) askRef.current(text)
    else setDraft(text)
  }, [])

  const speech = useSpeechRecognition({ locale, onTranscript: handleTranscript })

  useEffect(() => {
    readArray<ChatMessage>(HISTORY_KEY).then((stored) => {
      if (stored.length) dispatch({ type: 'hydrate', messages: stored })
      setHydrated(true)
    })
  }, [])

  useEffect(() => {
    if (!hydrated) return
    void writeJson(HISTORY_KEY, persistableMessages(conversation.messages))
  }, [conversation.messages, hydrated])

  const checkHealth = useCallback(async () => {
    try {
      const snapshot = await getHealth(baseUrl)
      setHealth(snapshot)
      setHealthTone(snapshot.status === 'healthy' ? 'ok' : 'warn')
      setHealthDetail(describeHostIssue(baseUrl))
    } catch (error) {
      setHealth(null)
      setHealthTone('danger')
      setHealthDetail(describeHostIssue(baseUrl) || describeError(error))
    }
  }, [baseUrl])

  const refreshDocument = useCallback(() => {
    void getDocument(baseUrl)
      .then((doc) => {
        if (!doc.loaded || !doc.filename) {
          setStats(null)
          return
        }
        setStats({
          filename: doc.filename,
          total_pages: doc.total_pages ?? 0,
          total_chunks: doc.total_chunks ?? 0,
          summary: doc.summary ?? '',
          index_ms: doc.index_ms,
          chunking: doc.chunking,
        })
      })
      .catch(() => undefined)
  }, [baseUrl])

  useEffect(() => {
    void checkHealth()
    void getVoices(baseUrl).then(setVoices).catch(() => setVoices([]))
    refreshDocument()
  }, [baseUrl, checkHealth, refreshDocument])

  // The card describes the file the server is holding, so when that changes underneath
  // it - another phone uploaded a PDF, the server restarted - the chip follows on the
  // next health poll instead of showing a count from a document that is gone.
  useEffect(() => {
    if (!health) return
    const loaded = (health.current_document ?? '').trim()
    if (!loaded && !stats) return
    if (loaded === (stats?.filename ?? '') && health.total_chunks === stats?.total_chunks) return
    refreshDocument()
  }, [health, stats, refreshDocument])

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state === 'active') void checkHealth()
    })
    return () => subscription.remove()
  }, [checkHealth])

  useEffect(() => {
    if (speech.error) setNotice(speech.error.message)
  }, [speech.error])

  useEffect(() => {
    if (player.error) setNotice(player.error)
  }, [player.error])

  const synthesize = useCallback(
    async (id: string, segments: string[], textTimings: Timings) => {
      if (!segments.length) {
        dispatch({ type: 'voice', id, patch: { audioUrls: null } })
        setNotice('This answer has no spoken copy, so there is nothing to synthesize.')
        return
      }

      const [head, ...tail] = segments
      const urls: string[] = []
      let ttsMs = 0

      // Both requests go out together: the head decides when speech starts, the
      // tail is already on its way and is appended to the queue when it lands.
      const tailPromise = tail.length ? speakSegments(baseUrl, tail, voiceId) : null
      if (tailPromise) tailPromise.catch(() => undefined)

      try {
        const first = await speakSegments(baseUrl, [head], voiceId)
        urls.push(...first.segments.map((entry) => entry.audio_url))
        ttsMs += first.timings?.tts_ms ?? 0
        if (!urls.length) throw new BackendError('parse', 'The server returned no audio for this answer.')
        const headNotice = partialSpeechNotice(first)
        dispatch({ type: 'clips', id, audioUrls: [...urls] })
        if (settings.autoSpeak && !settings.muted) player.playSegments(urls, id)
        if (headNotice) setNotice(headNotice)
      } catch (error) {
        dispatch({ type: 'voice', id, patch: { audioUrls: null } })
        setNotice("The answer is on screen, but the server could not speak it.")
        return
      }

      if (!tailPromise) {
        dispatch({
          type: 'voice',
          id,
          patch: { audioUrls: [...urls], timings: { ...textTimings, tts_ms: ttsMs } },
        })
        return
      }

      try {
        const rest = await tailPromise
        const restUrls = rest.segments.map((entry) => entry.audio_url)
        ttsMs += rest.timings?.tts_ms ?? 0
        player.enqueue(restUrls, id)
        dispatch({
          type: 'voice',
          id,
          patch: {
            audioUrls: [...urls, ...restUrls],
            voice: rest.voice ?? voiceId,
            timings: { ...textTimings, tts_ms: ttsMs },
          },
        })
        const tailNotice = partialSpeechNotice(rest)
        if (tailNotice) setNotice(tailNotice)
      } catch (error) {
        dispatch({ type: 'voice', id, patch: { audioUrls: [...urls], timings: { ...textTimings, tts_ms: ttsMs } } })
        setNotice("The answer is on screen and its first clip plays, but the rest could not be spoken.")
      }
    },
    [baseUrl, player, settings.autoSpeak, settings.muted, voiceId]
  )

  const ask = useCallback(
    async (raw: string, reuseId?: string) => {
      const question = raw.trim()
      if (!question || busy) return
      player.stop()
      setDraft('')
      setNotice(null)
      void Haptics.selectionAsync().catch(() => undefined)

      // Built before the question lands in state, so the exchange in flight is never
      // sent as history: the server reads the last real question out of this list.
      const history = askHistory(messagesRef.current)

      // `reuseId` means a failed answer is being retried in place, so the
      // question must not be appended a second time.
      let placeholderId: string
      if (reuseId) {
        placeholderId = reuseId
        dispatch({ type: 'retry', id: reuseId })
      } else {
        placeholderId = makeId('a')
        dispatch({
          type: 'ask',
          question: { id: makeId('q'), role: 'user', text: question, createdAt: Date.now() },
          placeholder: { id: placeholderId, role: 'assistant', text: '', createdAt: Date.now(), stage: 'working' },
        })
      }
      setBusy(true)
      const startedAt = Date.now()
      const controller = new AbortController()

      try {
        const data = await askText(
          baseUrl,
          { query: question, history },
          controller.signal
        )
        const timings: Timings = data.timings ?? {}
        dispatch({
          type: 'answer',
          id: placeholderId,
          patch: {
            text: data.answer,
            sources: data.sources ?? [],
            outOfContext: data.is_out_of_context,
            noDocument: data.no_document,
            voice: voiceId,
            latencyMs: Date.now() - startedAt,
            engine: data.engine,
            score: typeof data.retrieval_score === 'number' ? data.retrieval_score : undefined,
            timings,
          },
        })
        // The answer is readable now; speech must never hold the composer shut.
        setBusy(false)
        void checkHealth()
        await synthesize(placeholderId, data.spoken_segments ?? [], timings)
      } catch (error) {
        dispatch({
          type: 'fail',
          id: placeholderId,
          text: 'The server could not answer that. Check the connection and ask again.',
        })
        setHealthTone('danger')
      } finally {
        setBusy(false)
      }
    },
    [baseUrl, busy, checkHealth, player, voiceId, synthesize]
  )

  useEffect(() => {
    askRef.current = ask
  }, [ask])

  const retry = useCallback(
    (message: ChatMessage) => {
      const index = conversation.messages.findIndex((entry) => entry.id === message.id)
      const question = index > 0 ? conversation.messages[index - 1].text : ''
      if (!question) return
      void ask(question, message.id)
    },
    [conversation.messages, ask]
  )

  const pickDocument = useCallback(async () => {
    try {
      const picked = await DocumentPicker.getDocumentAsync({ type: 'application/pdf', copyToCacheDirectory: true })
      if (picked.canceled) return
      const asset = picked.assets[0]
      if (!asset) {
        setNotice('The picker returned no file.')
        return
      }
      setUploading(true)
      setNotice(null)
      player.stop()
      dispatch({
        type: 'system',
        message: {
          id: makeId('s'),
          role: 'system',
          text: `Uploading ${asset.name} and indexing it`,
          createdAt: Date.now(),
        },
      })
      const response = await uploadPdf(baseUrl, { uri: asset.uri, name: asset.name, mimeType: asset.mimeType })
      const stats = response.stats
      setStats(stats)
      const profile = stats.chunking
      const indexedIn =
        settings.telemetry && typeof stats.index_ms === 'number'
          ? ` in ${formatLatency(stats.index_ms) ?? `${stats.index_ms} ms`}`
          : ''
      const shape = settings.telemetry && profile?.chunks
        ? `, passages of ${profile.words_avg ?? '?'} words (shortest ${profile.words_min ?? '?'}, longest ${
            profile.words_max ?? '?'
          }, ${profile.headings_tracked ?? 0} headings carried)`
        : ''
      dispatch({
        type: 'system',
        message: {
          id: makeId('s'),
          role: 'system',
          text: `${stats.filename} indexed${indexedIn}: ${plural(stats.total_pages, 'page', 'pages')}, ${plural(
            stats.total_chunks,
            'chunk',
            'chunks'
          )}${shape}. Ask me anything about it.`,
          createdAt: Date.now(),
        },
      })
      if (settings.autoSpeak && !settings.muted && response.welcome_segments.length) {
        try {
          const spoken = await speakSegments(baseUrl, response.welcome_segments, voiceId)
          player.playSegments(
            spoken.segments.map((entry) => entry.audio_url),
            'welcome'
          )
          const welcomeNotice = partialSpeechNotice(spoken)
          if (welcomeNotice) setNotice(welcomeNotice)
        } catch (error) {
          setNotice("The document is indexed but the welcome message could not be spoken.")
        }
      }
      void checkHealth()
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => undefined)
    } catch (error) {
      dispatch({
        type: 'system',
        message: {
          id: makeId('s'),
          role: 'system',
          text: 'That document could not be opened. Choose another PDF.',
          createdAt: Date.now(),
          failed: true,
        },
      })
    } finally {
      setUploading(false)
    }
  }, [baseUrl, checkHealth, player, settings.autoSpeak, settings.muted, voiceId])

  const onTalkStart = useCallback(() => {
    if (!speech.availability.available) {
      setNotice(
        speech.availability.detail
          ? `Voice input is not available in this build: ${speech.availability.detail}`
          : 'Voice input is not available in this build. Tap the text field and use the microphone key on the keyboard.'
      )
      setFocusSignal((value) => value + 1)
      return
    }
    player.stop()
    void speech.start()
  }, [player, speech])

  const onTalkStop = useCallback(() => {
    if (speech.isListening) void Haptics.selectionAsync().catch(() => undefined)
    speech.stop()
  }, [speech])

  const orderedMessages = [...conversation.messages].reverse()

  const selectedVoiceName =
    voices.find((voice) => voice.id === voiceId)?.name ?? voiceId

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={[styles.header, { paddingTop: insets.top + spacing.sm }]}>
        <View style={styles.headerTop}>
          <StatusDot tone={healthTone} />
          <Text style={styles.host} numberOfLines={1}>
            {shortHost(baseUrl)}
          </Text>
          <Pressable onPress={onChangeServer} hitSlop={8} accessibilityRole="button" accessibilityLabel="Change server">
            <Text style={styles.headerAction}>Change</Text>
          </Pressable>
          <Pressable onPress={() => setShowVoices(true)} hitSlop={8} accessibilityRole="button" accessibilityLabel="Reply voice">
            <Text style={styles.headerAction}>Voice</Text>
          </Pressable>
          <Pressable onPress={() => setShowSettings(true)} hitSlop={8} accessibilityRole="button" accessibilityLabel="Settings">
            <Text style={styles.headerAction}>Settings</Text>
          </Pressable>
        </View>

        <Pressable
          onPress={uploading ? undefined : pickDocument}
          accessibilityRole="button"
          accessibilityLabel="Upload a PDF document"
          style={({ pressed }) => [styles.docBar, { opacity: pressed ? 0.85 : 1 }]}
        >
          <Text style={styles.docIcon}>PDF</Text>
          <Text style={styles.docText} numberOfLines={2}>
            {stats
              ? `${stats.filename} • ${plural(stats.total_pages, 'page', 'pages')} • ${plural(
                  stats.total_chunks,
                  'chunk',
                  'chunks'
                )}${settings.telemetry && stats.chunking?.words_p50 ? ` • ~${stats.chunking.words_p50} words per passage` : ''}`
              : health && health.current_document !== 'None'
                ? `${health.current_document} • ${plural(health.total_chunks, 'chunk', 'chunks')} indexed`
                : 'No document loaded — tap to choose a PDF'}
          </Text>
          <Text style={styles.docAction}>{uploading ? 'Indexing' : 'Upload'}</Text>
        </Pressable>
      </View>

      <FlatList
        data={orderedMessages}
        keyExtractor={(item) => item.id}
        inverted
        renderItem={({ item }) => (
          <MessageBubble
            message={item}
            isPlaying={player.activeId === item.id && player.isPlaying}
            playingSegment={player.activeId === item.id ? player.segment + 1 : 0}
            showTelemetry={settings.telemetry}
            onReplay={(message) => {
              if (player.activeId === message.id && player.isPlaying) player.stop()
              else player.playSegments(message.audioUrls, message.id)
            }}
            onRetry={retry}
          />
        )}
        ListEmptyComponent={
          <View style={styles.empty}>
            {documentName ? (
              <>
                <Text style={styles.emptyTitle}>Ask about {documentName}</Text>
                <Text style={styles.emptyBody}>
                  Tap the microphone at the bottom and speak, or type below. The passage is found on the server and the
                  answer comes back as speech.
                </Text>
              </>
            ) : (
              <>
                <Text style={styles.emptyTitle}>Voice answers from your own PDF</Text>
                <Text style={styles.emptyBody}>
                  Upload a document, then tap the microphone and ask. The server finds the passage, replies and speaks
                  the answer back to you.
                </Text>
                <Btn
                  label={uploading ? 'Indexing document' : 'Choose a PDF'}
                  onPress={pickDocument}
                  loading={uploading}
                />
              </>
            )}
            <View style={styles.starterWrap}>
              {STARTERS.map((prompt) => (
                <Chip key={prompt} label={prompt} onPress={() => ask(prompt)} />
              ))}
            </View>
          </View>
        }
        contentContainerStyle={styles.listContent}
        keyboardDismissMode="interactive"
      />

      <View style={{ paddingBottom: Math.max(insets.bottom, spacing.sm) }}>
        <Composer
          draft={draft}
          onDraftChange={setDraft}
          onSend={() => ask(draft)}
          onTalkStart={onTalkStart}
          onTalkStop={onTalkStop}
          onTalkCancel={speech.abort}
          phase={speech.phase}
          level={speech.level}
          listening={speech.isListening}
          busy={busy}
          voiceAvailable={speech.availability.available}
          notice={notice}
          onDismissNotice={() => {
            setNotice(null)
            speech.dismissError()
          }}
          autoFocusSignal={focusSignal}
        />
      </View>

      {healthTone === 'danger' && !notice ? (
        <Pressable style={styles.offlineBanner} onPress={checkHealth} accessibilityRole="button">
          <Text style={styles.offlineText}>Server unreachable. Tap to retry. {healthDetail ? `(${healthDetail})` : ''}</Text>
        </Pressable>
      ) : null}

      <VoiceSheet
        visible={showVoices}
        voices={voices}
        selected={voiceId}
        loading={voices.length === 0}
        serverDefault={health?.default_voice}
        listen={settings.listen}
        micLanguage={locale}
        onClose={() => setShowVoices(false)}
        onListen={(mode) => update({ listen: mode })}
        onSelect={(voice) => {
          update({ voiceId: voice.id })
          setShowVoices(false)
          setNotice(
            settings.listen === 'auto'
              ? `Voice set to ${voice.name}. Your next answer uses it — and the mic switches to ${voice.locale}.`
              : `Voice set to ${voice.name}. Your next answer uses it; the mic stays on ${locale} because you set it.`
          )
        }}
      />

      <SettingsSheet
        visible={showSettings}
        onClose={() => setShowSettings(false)}
        baseUrl={baseUrl}
        health={health}
        healthTone={healthTone}
        healthDetail={healthDetail}
        onRecheck={checkHealth}
        autoSpeak={settings.autoSpeak}
        muted={settings.muted}
        telemetry={settings.telemetry}
        onToggleAutoSpeak={(value) => update({ autoSpeak: value })}
        onToggleMute={(value) => update({ muted: value })}
        onToggleTelemetry={(value) => update({ telemetry: value })}
        speech={speech.availability}
        onOpenVoices={() => {
          setShowSettings(false)
          setShowVoices(true)
        }}
        voiceName={selectedVoiceName}
        micLanguage={locale}
        voiceNote={
          selectedVoice
            ? selectedVoice.local
              ? 'spoken on the server machine'
              : 'spoken over the network'
            : undefined
        }
        onClearConversation={() => {
          dispatch({ type: 'clear' })
          player.stop()
          void writeJson(HISTORY_KEY, [])
        }}
      />
    </KeyboardAvoidingView>
  )
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  header: { paddingHorizontal: spacing.md, paddingBottom: spacing.sm, backgroundColor: colors.surface },
  headerTop: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.sm },
  host: { color: colors.textMuted, fontSize: font.xs, flex: 1 },
  headerAction: { color: colors.accent, fontSize: font.xs, fontWeight: '700', paddingHorizontal: spacing.xs },
  docBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.surfaceAlt,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
  },
  docIcon: { color: colors.accentInk, backgroundColor: colors.accent, fontSize: 10, fontWeight: '800', paddingHorizontal: 6, paddingVertical: 3, borderRadius: 4 },
  docText: { color: colors.text, fontSize: font.sm, flex: 1, lineHeight: 19 },
  docAction: { color: colors.accent, fontSize: font.xs, fontWeight: '700' },
  listContent: { paddingVertical: spacing.md, flexGrow: 1, justifyContent: 'flex-start' },
  empty: { paddingHorizontal: spacing.xl, paddingTop: spacing.xl, gap: spacing.md },
  emptyTitle: { color: colors.text, fontSize: font.lg, fontWeight: '700' },
  emptyBody: { color: colors.textMuted, fontSize: font.sm, lineHeight: 21 },
  starterWrap: { flexDirection: 'row', flexWrap: 'wrap', marginTop: spacing.md },
  offlineBanner: {
    position: 'absolute',
    left: spacing.md,
    right: spacing.md,
    bottom: 150,
    backgroundColor: 'rgba(255, 107, 107, 0.14)',
    borderColor: colors.danger,
    borderWidth: 1,
    borderRadius: radius.sm,
    padding: spacing.md,
  },
  offlineText: { color: colors.danger, fontSize: font.sm, lineHeight: 19 },
})
