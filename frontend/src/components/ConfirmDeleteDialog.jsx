import { useEffect } from 'react'

/* Confirmation for the two delete actions that share one rule: a lecture or a
   topic that no student has used is deleted outright, and one with student
   history is archived instead so no learning record is ever destroyed.

   The backend decides which of the two happens, from the data rather than
   from a flag, and returns the preview this dialog renders. Showing the
   teacher the mode AND the records being kept is the point — a generic "are
   you sure?" would hide the only thing worth knowing. */

const HISTORY_LABELS = [
  ['sessions', 'TeachBack session'],
  ['quiz_attempts', 'knowledge-check attempt'],
  ['activity_completions', 'completed activity'],
  ['observations', 'learning-state record'],
]

export default function ConfirmDeleteDialog({ preview, noun = 'lecture', busy, onCancel, onConfirm }) {
  // Escape closes the dialog, and so does clicking the backdrop — a modal a
  // user cannot dismiss is worse than no modal.
  useEffect(() => {
    if (!preview) return undefined
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [preview, busy, onCancel])

  if (!preview) return null
  const archiving = preview.mode === 'archive'
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-charcoal/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`${archiving ? 'Archive' : 'Delete'} ${noun} confirmation`}
      onClick={(e) => { if (e.target === e.currentTarget && !busy) onCancel() }}
    >
      <div className="card max-w-lg w-full p-6 space-y-4 max-h-[85vh] overflow-y-auto">
        <div className="text-lg font-bold text-charcoal">
          {archiving ? `Archive this ${noun}?` : `Delete this ${noun}?`}
        </div>
        <p className="text-sm text-charcoal-light">{preview.message}</p>
        {archiving && (
          <ul className="text-xs text-charcoal-light bg-zinc-50 border border-zinc-200 rounded p-3 space-y-1">
            {HISTORY_LABELS
              .filter(([key]) => preview.history?.[key] > 0)
              .map(([key, label]) => (
                <li key={key}>
                  {preview.history[key]} {label}{preview.history[key] === 1 ? '' : 's'} — <strong>kept</strong>
                </li>
              ))}
          </ul>
        )}
        <div className="flex justify-end gap-3">
          <button onClick={onCancel} className="btn-secondary" disabled={busy}>Cancel</button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className="px-4 py-2 rounded-md font-semibold text-white bg-brand hover:bg-brand-dark disabled:opacity-50"
          >
            {busy ? 'Working…' : archiving ? `Archive ${noun}` : `Delete ${noun}`}
          </button>
        </div>
      </div>
    </div>
  )
}
