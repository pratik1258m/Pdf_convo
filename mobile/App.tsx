import { useEffect, useState } from 'react'
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native'
import { StatusBar } from 'expo-status-bar'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import { ChatScreen } from './src/components/ChatScreen'
import { ConnectScreen } from './src/components/ConnectScreen'
import { useSettings } from './src/state/settings'
import { colors, font } from './src/theme'

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <Root />
    </SafeAreaProvider>
  )
}

function Root() {
  const { settings, ready, update, rememberUrl } = useSettings()
  const [server, setServer] = useState<string | null>(null)

  useEffect(() => {
    if (ready && settings.baseUrl) setServer(settings.baseUrl)
  }, [ready, settings.baseUrl])

  if (!ready) {
    return (
      <View style={styles.splash}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.splashText}>Restoring your session</Text>
      </View>
    )
  }

  if (!server) {
    return (
      <ConnectScreen
        settings={settings}
        onConnect={(url) => {
          rememberUrl(url)
          setServer(url)
        }}
      />
    )
  }

  return (
    <ChatScreen
      baseUrl={server}
      settings={settings}
      update={update}
      onChangeServer={() => setServer(null)}
    />
  )
}

const styles = StyleSheet.create({
  splash: { flex: 1, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center', gap: font.sm },
  splashText: { color: colors.textMuted, fontSize: font.sm },
})
