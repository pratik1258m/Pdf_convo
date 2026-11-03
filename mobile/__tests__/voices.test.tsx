import { fireEvent, render } from '@testing-library/react-native'
import { SafeAreaProvider } from 'react-native-safe-area-context'

// The sheet file also hosts the recognition diagnostics panel, which reaches for
// a native module that a unit test has no reason to own.
jest.mock('expo-speech-recognition', () => ({
  ExpoSpeechRecognitionModule: {
    isRecognitionAvailable: () => true,
    supportsOnDeviceRecognition: () => false,
    supportsRecording: () => true,
    getSpeechRecognitionServices: () => [],
    getSupportedLocales: () => Promise.resolve({ installedLocales: [], requestedLocales: [] }),
  },
}))

import { VoiceSheet } from '../src/components/Sheets'
import type { ListenMode } from '../src/utils/format'
import type { VoiceOption } from '../src/types'

const VOICES: VoiceOption[] = [
  {
    id: 'supertonic-M1',
    name: 'Supertonic M1 (local - Hindi and English - Male)',
    locale: 'en-IN',
    gender: 'Male',
    recommended: true,
    engine: 'supertonic',
    local: true,
    available: true,
    reason: '',
  },
  {
    id: 'hi_IN-pratham-medium',
    name: 'Pratham (local - Hindi / Hinglish - Male)',
    locale: 'hi-IN',
    gender: 'Male',
    recommended: false,
    engine: 'piper',
    local: true,
    available: false,
    reason: 'hi_IN-pratham-medium.onnx is missing. Run: python3 -m piper.download_voices hi_IN-pratham-medium',
  },
  {
    id: 'en-IN-PrabhatNeural',
    name: 'Prabhat (Edge - Indian English - Male)',
    locale: 'en-IN',
    gender: 'Male',
    recommended: true,
    engine: 'edge',
    local: false,
    available: true,
    reason: '',
  },
]

async function renderSheet(
  onSelect = () => undefined,
  onListen = () => undefined,
  listen: ListenMode = 'auto'
) {
  return await render(
    <SafeAreaProvider
      initialMetrics={{
        frame: { x: 0, y: 0, width: 390, height: 844 },
        insets: { top: 0, left: 0, right: 0, bottom: 0 },
      }}
    >
      <VoiceSheet
        visible
        voices={VOICES}
        selected="supertonic-M1"
        loading={false}
        serverDefault="supertonic-M1"
        listen={listen}
        micLanguage={listen === 'auto' ? 'en-IN' : listen}
        onClose={() => undefined}
        onSelect={onSelect}
        onListen={onListen}
      />
    </SafeAreaProvider>
  )
}

describe('reply voice sheet', () => {
  // Each test queries its own render rather than `screen`: the sheet is a Modal,
  // and a stale container from the previous test is a tree that no longer exists.
  it('says where each voice is made, so a fast answer and a slow one are explainable', async () => {
    const { getByText } = await renderSheet()
    expect(getByText(/Supertonic M1/)).toBeTruthy()
    expect(getByText(/on the server • server default/)).toBeTruthy()
    expect(getByText(/over the network/)).toBeTruthy()
  })

  it('shows the reason a voice cannot be used instead of hiding it', async () => {
    const { getByText } = await renderSheet()
    expect(getByText(/hi_IN-pratham-medium\.onnx is missing/)).toBeTruthy()
  })

  it('lets the microphone language be set apart from the reply voice', async () => {
    const onListen = jest.fn()
    const { getByText } = await renderSheet(() => undefined, onListen)
    // Supertonic answers in both languages, so listening in Hindi has to be a
    // choice of its own rather than something the voice id has to imply.
    // RNTL 14 fires events through act(), so pressing without awaiting it leaves
    // the next test rendering into a tree React has not finished mounting.
    await fireEvent.press(getByText('हिन्दी'))
    expect(onListen).toHaveBeenCalledWith('hi-IN')

    await fireEvent.press(getByText('English'))
    expect(onListen).toHaveBeenCalledWith('en-IN')
  })

  it('names the language the microphone is actually listening in', async () => {
    const { getByText } = await renderSheet(() => undefined, () => undefined, 'hi-IN')
    expect(getByText(/Listening in hi-IN/)).toBeTruthy()
  })

  it('refuses to select a voice this machine cannot speak', async () => {
    const onSelect = jest.fn()
    const { getByText } = await renderSheet(onSelect)
    await fireEvent.press(getByText(/Pratham \(local/))
    expect(onSelect).not.toHaveBeenCalled()

    await fireEvent.press(getByText(/Prabhat \(Edge/))
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: 'en-IN-PrabhatNeural' }))
  })
})
