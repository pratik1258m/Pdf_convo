# Convo PDF (React Native client)

Expo SDK 58 / React Native 0.88 / React 19 / TypeScript. Talk to a PDF that lives on your own machine: tap the microphone, ask, and the answer appears as text within milliseconds and starts being spoken as soon as the first sentence exists. The mic is a switch, not a push-to-talk key: one tap starts listening, the next tap sends what it heard, and Cancel discards the phrase.

## Why a development build

The microphone, the document picker and the audio queue are third-party native modules (`expo-speech-recognition`, `expo-document-picker`, `expo-audio`). Expo Go does not ship them, so voice input needs a dev build:

```bash
cd /Users/pratik/Documents/realtime_convo/mobile
npm install
npx expo start --android       # or: npx expo start --ios
```

Press `a` / `i` in the Metro terminal and follow the prompt to install the build on the device, or produce an APK with the Android project:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 npx expo run:android --variant apk
```

## Point it at the server

Start the backend first (see the repository README), then enter its LAN address on the connect screen, for example `http://10.151.207.43:8080`.

- `localhost` on a phone is the phone, not your Mac. The app says so when you try.
- On an Android emulator use `http://10.0.2.2:8080`.
- Cleartext HTTP is allowed in `app.json` (`usesCleartextTraffic`, `NSAllowsLocalNetworking`); the phone and the computer must be on the same network.

## How a question flows

1. `POST /api/ask_text` with the question. The answer, its page citations, the retrieval score, the engine label and the server's own stage timings come back in a couple of milliseconds, and the answer is rendered immediately. In that response, `sources` is the list of cited page numbers (empty for a refusal), `retrieval_score` is how much of the question the index recognised, `confidence_score` is the same number for an answered question and `0.0` for a refusal, and `is_out_of_context` marks an answer that cites nothing in the document. The composer unlocks at that point; speech never blocks typing.
2. The spoken copy arrives already split into segments. `POST /api/speak` is called twice, at the same moment, once for the first sentence and once for the rest; the rest go out in batches of twelve segments, one batch after another, because the local engines make one clip at a time.
3. The first clip plays the moment it lands. The rest are appended to the same queue by `useReplyPlayer.enqueue`, so a long answer keeps talking instead of stopping after one sentence. Press Stop and the queue is cancelled; late clips for that answer are dropped.
4. Every repeated answer is a disk-cache hit on the server (about 1 ms), so re-asking is nearly instant.

Transcription runs on the phone's native speech recogniser through `expo-speech-recognition`; the app does not force it offline — whether recognition is on-device depends on the languages installed on the handset, and Settings reports what the device answers. The recognition locale follows the reply voice you pick, so choosing a Hindi voice switches the microphone to `hi-IN` as well.

## Voice, and where it is made

The voice sheet is filled from `GET /api/voices`, one row per voice, and each row says what the phone cannot otherwise know: whether the audio is made on the server machine or fetched over the network, the voice's locale and gender, and which one the server chose as its default. The engine is named in each voice's display name, and the response itself carries the `engine` and the `format` the audio will get (`wav` for the local engines, `mp3` for Edge).

- Nothing is hardcoded as selectable. A voice whose model is missing on the server comes back from `/api/voices` with `available: false`, is rendered greyed out as not available on this server, and tapping it does nothing — the sheet deliberately does not print the server's internal reason. An answer cannot be promised and then left silent. A machine with nothing installed still offers a working voice, because the server's default is whichever engine it can actually run.
- The reply voice decides what the microphone hears, unless you override it. `micLocale()` derives the recognition language from the selected voice, so `hi_IN-pratham-medium` puts the mic in `hi-IN` and `en_US-ryan-medium` puts it in `en-US`. Supertonic speaks both languages from one voice, so the voice sheet's "Microphone language" row is what chooses the listening language (`auto`, English, or हिन्दी) rather than the voice id guessing it. An id shaped like neither gets the `en-IN` default instead of a string the recognizer would reject.
- The audio is a WAV at 22.05 kHz (Piper) or 44.1 kHz (Supertonic) instead of an MP3 whenever a local voice is chosen; `expo-audio` plays both from the same `/audio/...` URL. The reply bubble labels the voice from the `voice` field on the `/api/speak` response, and `speech_engine` in that same response names the engine that made the clips.
- One answer, one engine. The voice is sent with both speech requests, so a reply is never half local and half network.

## What you see on screen

- The document bar shows the indexed file with its page count and chunk count; the median passage size is added there only when the developer switch is on. The numbers come from the upload response and `GET /api/document`, with `GET /api/health` covering the bar until they arrive.
- Each answer bubble shows the answer, the page chips it came from and Replay/Retry controls. Stage timings, the match score and the engine or voice behind a reply are true measurements but stay hidden until Settings → Developer → "Show engine and timings" is on: a caller listening to an answer has no use for them, and a model name in the bubble makes the reply read like a system report.
- An answer the document does not support is labelled "Outside this document" instead of being dressed up as grounded.
- The honest states are shown as they are: `partial` from `/api/speak` says how many sentences could not be spoken and why, `no_document` from `/api/ask_text` shows "No document loaded" rather than a not-found message, and a clip that no longer exists on the server is skipped with "N of M clips could not be played" while the rest of the answer keeps talking.
- Settings shows the chunking profile the server measured (median and range of passage words, headings carried, short runs merged, percentage of page text indexed) once the developer switch is on, and the device's speech-recognition capabilities, read fresh each time the sheet is opened.

## Layout

```
src/api/backend.ts        typed fetch wrapper, per-endpoint timeouts, BackendError kinds
src/api/normalize.ts      address validation, audio URL resolution, localhost detection
src/audio/speech.ts       native recognition hook (phase, level, availability, errors)
src/audio/playback.ts     sequential clip queue with late-arrival append and cancel
src/state/conversation.ts reducer for ask -> answer -> clips -> voice, AsyncStorage history
src/components/           ConnectScreen, ChatScreen, Composer, MessageBubble, Sheets, ui
```

## Checks

```bash
npm run typecheck   # tsc --noEmit
npm test            # jest + jest-expo, 45 tests
npx expo export --platform android --platform ios
```

The tests cover the reducer transitions, the exchange history that rides with a question (only completed pairs, the last four, the turn in flight left out), the server-address validation including the loopback warning, the playback queue (first clip starts before the rest exist, one advance per finished clip, Stop is not resumed, stale clips are ignored, a clip that fails to open is skipped and the rest of the answer still plays), the answer bubble copy including the "No document loaded" chip, the latency formatting, which language the microphone ends up in for each kind of voice id, and the voice sheet (a voice the server cannot speak is refused without printing the server's internals, and the listening language can be set apart from the reply voice).

`fireEvent` in React Native Testing Library 14 dispatches inside `act()` and returns a promise: a press that is not awaited leaves the next test rendering into a tree React has not finished mounting, and the assertions then fail against components that are on screen. Every press in these tests is awaited.

**Status of verification:** typechecked, unit-tested and bundle-exported on this machine. A Redmi `2312DRAABI` (Android, `gold_in_global`) was attached over USB: the release APK was installed with `adb install -r`, launched, and reached the server through both `adb reverse tcp:8080` and the ngrok tunnel. What the phone's own screen reports is still the user's observation, because HyperOS blocks injected input events (`adb shell input` needs "USB debugging (Security settings)"), so taps, microphone capture and the document picker on hardware cannot be driven from here.
