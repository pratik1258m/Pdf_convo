import { ActivityIndicator, Modal, Pressable, StyleSheet, Text, View, type ViewStyle } from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { colors, font, radius, spacing } from '../theme'

type ButtonVariant = 'primary' | 'soft' | 'ghost' | 'danger'

export function Btn({
  label,
  onPress,
  variant = 'primary',
  disabled,
  loading,
  style,
  accessibilityLabel,
}: {
  label: string
  onPress?: () => void
  variant?: ButtonVariant
  disabled?: boolean
  loading?: boolean
  style?: ViewStyle
  accessibilityLabel?: string
}) {
  const palette: Record<ButtonVariant, { bg: string; fg: string; border: string }> = {
    primary: { bg: colors.accent, fg: colors.accentInk, border: colors.accent },
    soft: { bg: colors.surfaceAlt, fg: colors.text, border: colors.border },
    ghost: { bg: 'transparent', fg: colors.textMuted, border: 'transparent' },
    danger: { bg: 'transparent', fg: colors.danger, border: colors.danger },
  }
  const tone = palette[variant]
  const inactive = disabled || loading

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={{ disabled: !!inactive }}
      onPress={inactive ? undefined : onPress}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: tone.bg, borderColor: tone.border, opacity: inactive ? 0.5 : pressed ? 0.82 : 1 },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator size="small" color={tone.fg} />
      ) : (
        <Text style={[styles.buttonText, { color: tone.fg }]} numberOfLines={1}>
          {label}
        </Text>
      )}
    </Pressable>
  )
}

export function Chip({
  label,
  tone = 'default',
  onPress,
}: {
  label: string
  tone?: 'default' | 'accent' | 'warn' | 'ok'
  onPress?: () => void
}) {
  const tones = {
    default: { bg: colors.surfaceHi, fg: colors.textMuted, border: colors.border },
    accent: { bg: colors.accentSoft, fg: colors.accent, border: colors.accent },
    warn: { bg: 'rgba(242, 169, 59, 0.14)', fg: colors.warn, border: colors.warn },
    ok: { bg: 'rgba(53, 196, 140, 0.14)', fg: colors.ok, border: colors.ok },
  } as const
  const tone2 = tones[tone]

  return (
    <Pressable
      accessibilityRole={onPress ? 'button' : undefined}
      onPress={onPress}
      style={({ pressed }) => [
        styles.chip,
        { backgroundColor: tone2.bg, borderColor: tone2.border, opacity: pressed && onPress ? 0.75 : 1 },
      ]}
    >
      <Text style={[styles.chipText, { color: tone2.fg }]}>{label}</Text>
    </Pressable>
  )
}

export function StatusDot({ tone }: { tone: 'ok' | 'warn' | 'danger' | 'idle' }) {
  const map = { ok: colors.ok, warn: colors.warn, danger: colors.danger, idle: colors.textFaint } as const
  return <View style={[styles.dot, { backgroundColor: map[tone] }]} accessibilityLabel={`status ${tone}`} />
}

export function Sheet({
  title,
  subtitle,
  visible,
  onClose,
  children,
}: {
  title: string
  subtitle?: string
  visible: boolean
  onClose: () => void
  children: React.ReactNode
}) {
  const insets = useSafeAreaInsets()

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.sheetScrim}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} accessibilityLabel="Close sheet" />
        <View
          style={[styles.sheet, { paddingBottom: insets.bottom + spacing.lg }]}
          accessibilityViewIsModal
        >
          <View style={styles.sheetHandle} />
          <Text style={styles.sheetTitle}>{title}</Text>
          {subtitle ? <Text style={styles.sheetSubtitle}>{subtitle}</Text> : null}
          <View style={styles.sheetBody}>{children}</View>
        </View>
      </View>
    </Modal>
  )
}

export function SectionTitle({ children }: { children: string }) {
  return <Text style={styles.sectionTitle}>{children}</Text>
}

const styles = StyleSheet.create({
  button: {
    minHeight: 46,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  buttonText: { fontSize: font.md, fontWeight: '600' },
  chip: {
    paddingHorizontal: spacing.sm + 2,
    minHeight: 26,
    borderRadius: radius.pill,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.xs + 2,
    marginBottom: spacing.xs,
  },
  chipText: { fontSize: font.xs, fontWeight: '600' },
  dot: { width: 8, height: 8, borderRadius: 4 },
  sheetScrim: { flex: 1, backgroundColor: colors.scrim, justifyContent: 'flex-end' },
  sheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: radius.lg,
    borderTopRightRadius: radius.lg,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    maxHeight: '82%',
    borderTopWidth: 1,
    borderColor: colors.border,
  },
  sheetHandle: {
    alignSelf: 'center',
    width: 42,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.border,
    marginBottom: spacing.md,
  },
  sheetTitle: { color: colors.text, fontSize: font.xl, fontWeight: '700' },
  sheetSubtitle: { color: colors.textMuted, fontSize: font.sm, marginTop: spacing.xs + 2, lineHeight: 20 },
  sheetBody: { marginTop: spacing.lg },
  sectionTitle: {
    color: colors.textFaint,
    fontSize: font.xs,
    fontWeight: '700',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
})
