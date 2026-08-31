import { useEffect, useState } from 'react'
import ConfirmDeleteDialog from '../components/ConfirmDeleteDialog.jsx'
import { TeacherContextBar, useTeacherContext } from '../components/TeacherContext.jsx'
import { api } from '../services/api.js'

const emptyTopic = {
  name: '',
  description: '',
  reference_explanation: '',
  opening_prompt: '',
  extension_question: '',
  concepts: [],
  relationships: [],
  misconceptions: [],
  activities: [],
}

const STATE_OPTIONS = [
  ['not_trying', 'Not Trying'],
  ['unclear', 'Unclear'],
  ['struggling', 'Struggling but Trying'],
  ['understanding', 'Understanding'],
  ['confident', 'Confident'],
]

export default function Topics() {
  const context = useTeacherContext()
  const subjectId = context.subject?.id
  const [topics, setTopics] = useState(null)
  const [archived, setArchived] = useState([])
  const [showArchived, setShowArchived] = useState(false)
  const [editing, setEditing] = useState(null) // {id?, ...topicData}
  const [deleting, setDeleting] = useState(null) // delete-preview being confirmed
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)

  /* Always re-read from the backend. SQLite is the source of truth and
     nothing about a topic is cached in the browser, so a delete or a restore
     is reflected by asking again rather than by editing local state. */
  const load = () => {
    if (!subjectId) return
    api.topics(subjectId).then(setTopics).catch((e) => setMessage({ kind: 'error', text: e.message }))
    api.topics(subjectId, true)
      .then((all) => setArchived(all.filter((t) => t.archived)))
      .catch(() => setArchived([]))
  }
  useEffect(() => { load() }, [subjectId]) // eslint-disable-line react-hooks/exhaustive-deps

  const openTopic = async (id) => {
    setMessage(null)
    const t = await api.topic(id)
    setEditing(t)
  }

  const save = async () => {
    setBusy(true)
    setMessage(null)
    try {
      if (editing.id) await api.updateTopic(editing.id, editing)
      else await api.createTopic(editing)
      setMessage({ kind: 'ok', text: 'Topic saved.' })
      setEditing(null)
      load()
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  /* The backend decides delete vs archive from the data and returns what it
     will do; the dialog shows the teacher that decision before it happens. */
  const askDelete = async (id) => {
    setMessage(null)
    try {
      setDeleting(await api.topicDeletePreview(id))
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    }
  }

  const confirmDelete = async () => {
    setBusy(true)
    try {
      const r = await api.deleteTopic(deleting.topic_id)
      setDeleting(null)
      setMessage({ kind: 'ok', text: r.message })
      if (editing?.id === r.topic_id) setEditing(null)
      load()
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  const restore = async (id) => {
    setMessage(null)
    try {
      await api.restoreTopic(id)
      setMessage({ kind: 'ok', text: 'Topic restored to your active list.' })
      load()
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    }
  }

  const setField = (k, v) => setEditing((t) => ({ ...t, [k]: v }))
  const setItem = (listKey, idx, k, v) =>
    setEditing((t) => {
      const list = [...t[listKey]]
      list[idx] = { ...list[idx], [k]: v }
      return { ...t, [listKey]: list }
    })
  const addItem = (listKey, item) => setEditing((t) => ({ ...t, [listKey]: [...t[listKey], item] }))
  const removeItem = (listKey, idx) =>
    setEditing((t) => ({ ...t, [listKey]: t[listKey].filter((_, i) => i !== idx) }))

  if (!editing) {
    return (
      <div className="space-y-5">
        <div className="banner">Topic Management</div>
        <TeacherContextBar context={context} />
        {message && (
          <div className={`card p-4 text-sm ${message.kind === 'ok' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-red-700 bg-red-50 border-red-200'}`}>
            {message.text}
          </div>
        )}
        <div className="flex justify-end">
          <button onClick={() => setEditing({ ...emptyTopic, subject_id: subjectId })} className="btn-primary">+ New topic</button>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {!topics && <div className="text-sm text-charcoal-light">Loading topics…</div>}
          {topics?.length === 0 && (
            <div className="card p-5 text-sm text-charcoal-light">
              No topics yet for <strong>{context.subject?.name}</strong>. Use
              <em> + New topic</em> to add one — or switch subject above.
            </div>
          )}
          {topics?.map((t) => (
            <div key={t.id} className="card p-5 flex flex-col hover:border-brand hover:shadow-md transition-all group">
              {/* the card body opens the editor; the actions below are real
                  buttons so Delete is never a hidden corner icon */}
              <button
                onClick={() => openTopic(t.id)}
                aria-label={`Open ${t.name} for editing`}
                className="text-left w-full flex-1"
              >
                <div className="font-bold text-charcoal group-hover:text-brand break-words">{t.name}</div>
                <p className="text-sm text-charcoal-light mt-1 line-clamp-2">{t.description}</p>
                <div className="text-xs text-charcoal-light mt-3">
                  {t.concept_count} concepts · {t.relationship_count ?? 0} relationships · {t.misconception_count} misconceptions · {t.activity_count} activities
                </div>
              </button>
              <div className="flex items-center justify-between gap-3 mt-4 pt-3 border-t border-zinc-100">
                <button
                  onClick={() => openTopic(t.id)}
                  aria-label={`Edit topic ${t.name}`}
                  className="text-sm font-semibold text-brand hover:underline"
                >
                  Edit →
                </button>
                <button
                  onClick={() => askDelete(t.id)}
                  aria-label={`Delete topic ${t.name}`}
                  title={`Delete ${t.name}`}
                  className="text-sm font-semibold text-brand border border-brand/40 hover:bg-brand hover:text-white rounded-md px-3 py-1.5 transition-colors"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        {archived.length > 0 && (
          <div className="card">
            <button
              onClick={() => setShowArchived((v) => !v)}
              className="card-header w-full text-left normal-case"
            >
              <span>Archived topics ({archived.length})</span>
              <span className="text-white/80 font-normal">{showArchived ? 'hide' : 'show'}</span>
            </button>
            {showArchived && (
              <div className="p-4 space-y-2">
                <p className="text-xs text-charcoal-light">
                  These topics were removed from the active list because students had already
                  worked on them. No new TeachBack can start on them, and all their existing
                  sessions, knowledge-check attempts and progress records were kept.
                </p>
                {archived.map((t) => (
                  <div key={t.id} className="flex items-center justify-between gap-3 border border-zinc-200 rounded-md p-3">
                    <div className="min-w-0">
                      <div className="font-semibold text-charcoal truncate">{t.name}</div>
                      <div className="text-xs text-charcoal-light">
                        Archived{t.archived_at ? ` on ${t.archived_at.slice(0, 10)}` : ''}
                      </div>
                    </div>
                    <button onClick={() => restore(t.id)} className="btn-secondary whitespace-nowrap">Restore</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <ConfirmDeleteDialog
          noun="topic"
          preview={deleting}
          busy={busy}
          onCancel={() => setDeleting(null)}
          onConfirm={confirmDelete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div className="banner">{editing.id ? `Edit Topic — ${editing.name}` : 'New Topic'}</div>
      {message?.kind === 'error' && (
        <div className="card p-4 text-sm text-red-700 bg-red-50 border-red-200">{message.text}</div>
      )}

      <div className="card p-5 space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="label">Topic name</label>
            <input className="input" value={editing.name} onChange={(e) => setField('name', e.target.value)} />
          </div>
          <div>
            <label className="label">Short description</label>
            <input className="input" value={editing.description} onChange={(e) => setField('description', e.target.value)} />
          </div>
        </div>
        <div>
          <label className="label">Reference explanation (a short model answer, written in plain language)</label>
          <textarea className="input min-h-[80px]" value={editing.reference_explanation} onChange={(e) => setField('reference_explanation', e.target.value)} />
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="label">Opening prompt (optional)</label>
            <input className="input" value={editing.opening_prompt} onChange={(e) => setField('opening_prompt', e.target.value)} placeholder="Teach me what you understand about…" />
          </div>
          <div>
            <label className="label">Extension question (asked when everything is covered)</label>
            <input className="input" value={editing.extension_question} onChange={(e) => setField('extension_question', e.target.value)} />
          </div>
        </div>
      </div>

      {/* concepts */}
      <div className="card">
        <div className="card-header">
          <span>Required concepts</span>
          <button onClick={() => addItem('concepts', { name: '', description: '', main_question: '', easier_question: '', probe_question: '', application_question: '' })} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          <p className="text-xs text-charcoal-light">
            <strong>What important ideas should students understand?</strong> Each concept gets its own short
            questions in the TeachBack conversation. Students don&apos;t need to use these exact words — any
            correct phrasing of the idea counts.
          </p>
          {editing.concepts.length === 0 && <p className="text-sm text-charcoal-light">No concepts yet — add the ideas a good explanation must contain.</p>}
          {editing.concepts.map((c, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-2 gap-3 relative">
              <div>
                <label className="label">Name</label>
                <input className="input" value={c.name} onChange={(e) => setItem('concepts', i, 'name', e.target.value)} />
              </div>
              <div>
                <label className="label">Description (what a correct explanation should express)</label>
                <input className="input" value={c.description} onChange={(e) => setItem('concepts', i, 'description', e.target.value)} />
              </div>
              <div>
                <label className="label">Main question (asked first)</label>
                <input className="input" value={c.main_question || ''} onChange={(e) => setItem('concepts', i, 'main_question', e.target.value)} />
              </div>
              <div>
                <label className="label">Easier question (if the answer is unclear)</label>
                <input className="input" value={c.easier_question || ''} onChange={(e) => setItem('concepts', i, 'easier_question', e.target.value)} />
              </div>
              <div>
                <label className="label">Probe question (if partly right)</label>
                <input className="input" value={c.probe_question} onChange={(e) => setItem('concepts', i, 'probe_question', e.target.value)} />
              </div>
              <div>
                <label className="label">Application question (optional stretch)</label>
                <input className="input" value={c.application_question || ''} onChange={(e) => setItem('concepts', i, 'application_question', e.target.value)} />
              </div>
              <button onClick={() => removeItem('concepts', i)} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      {/* concept relationships */}
      <div className="card">
        <div className="card-header">
          <span>Concept relationships</span>
          <button onClick={() => addItem('relationships', { source: '', label: 'relates to', target: '', description: '', contradiction: '', probe_question: '' })} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          <p className="text-xs text-charcoal-light">
            <strong>How do these ideas connect?</strong> (e.g. Backpropagation → computes → Gradient.)
            TeachBack uses the correct sentence to recognise when a student expresses the connection,
            and the optional incorrect example to notice when a student has the connection wrong.
          </p>
          {(editing.relationships || []).length === 0 && (
            <p className="text-sm text-charcoal-light">No relationships yet — add the key connections a good explanation should make.</p>
          )}
          {(editing.relationships || []).map((r, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-3 gap-3 relative">
              <div>
                <label className="label">Source concept</label>
                <input className="input" value={r.source} onChange={(e) => setItem('relationships', i, 'source', e.target.value)} placeholder="Gradient descent" />
              </div>
              <div>
                <label className="label">Link (verb)</label>
                <input className="input" value={r.label} onChange={(e) => setItem('relationships', i, 'label', e.target.value)} placeholder="uses" />
              </div>
              <div>
                <label className="label">Target concept</label>
                <input className="input" value={r.target} onChange={(e) => setItem('relationships', i, 'target', e.target.value)} placeholder="Gradient" />
              </div>
              <div className="md:col-span-3">
                <label className="label">Correct sentence (how the connection should be explained)</label>
                <input className="input" value={r.description} onChange={(e) => setItem('relationships', i, 'description', e.target.value)} placeholder="Gradient descent uses the gradients to decide how to change the weights." />
              </div>
              <div className="md:col-span-2">
                <label className="label">Common incorrect explanation (optional)</label>
                <input className="input" value={r.contradiction || ''} onChange={(e) => setItem('relationships', i, 'contradiction', e.target.value)} placeholder="Gradient descent updates the weights so the loss increases." />
                <p className="text-[11px] text-charcoal-light mt-1">
                  An example of how a student might incorrectly explain this connection. TeachBack uses it to spot possible misconceptions.
                </p>
              </div>
              <div>
                <label className="label">Probe question (asked if the link is missing/confused)</label>
                <input className="input" value={r.probe_question || ''} onChange={(e) => setItem('relationships', i, 'probe_question', e.target.value)} />
              </div>
              <button onClick={() => removeItem('relationships', i)} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      {/* misconceptions */}
      <div className="card">
        <div className="card-header">
          <span>Known misconceptions</span>
          <button onClick={() => addItem('misconceptions', { name: '', description: '', clarification: '', probe_question: '' })} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          <p className="text-xs text-charcoal-light">
            <strong>What do students commonly get wrong?</strong> When a student&apos;s answer matches a wrong
            claim, TeachBack explains the distinction and asks them to try again.
          </p>
          {editing.misconceptions.length === 0 && <p className="text-sm text-charcoal-light">No misconceptions yet — add wrong beliefs students commonly express.</p>}
          {editing.misconceptions.map((m, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-2 gap-3 relative">
              <div>
                <label className="label">Name (shown to student)</label>
                <input className="input" value={m.name} onChange={(e) => setItem('misconceptions', i, 'name', e.target.value)} />
              </div>
              <div>
                <label className="label">Wrong claim (as a student would phrase it)</label>
                <input className="input" value={m.description} onChange={(e) => setItem('misconceptions', i, 'description', e.target.value)} />
              </div>
              <div>
                <label className="label">Correct clarification (contrast sentence)</label>
                <input className="input" value={m.clarification} onChange={(e) => setItem('misconceptions', i, 'clarification', e.target.value)} />
              </div>
              <div>
                <label className="label">Probe question (asked if detected)</label>
                <input className="input" value={m.probe_question} onChange={(e) => setItem('misconceptions', i, 'probe_question', e.target.value)} />
              </div>
              <button onClick={() => removeItem('misconceptions', i)} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      {/* activities */}
      <div className="card">
        <div className="card-header">
          <span>Activities (one per learning state)</span>
          <button onClick={() => addItem('activities', { title: '', description: '', kind: 'practice', target_state: 'understanding', content: '', question: '' })} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          <p className="text-xs text-charcoal-light">
            <strong>What should students do at different learning states?</strong> After a session, the student
            is recommended the activity matching their estimated state — and can open and complete it directly.
            Add the content students will actually see and the short task they should complete.
          </p>
          {editing.activities.map((a, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-2 gap-3 relative">
              <div>
                <label className="label">Title</label>
                <input className="input" value={a.title} onChange={(e) => setItem('activities', i, 'title', e.target.value)} />
              </div>
              <div>
                <label className="label">Description</label>
                <input className="input" value={a.description} onChange={(e) => setItem('activities', i, 'description', e.target.value)} />
              </div>
              <div>
                <label className="label">Kind (shown as the activity tag)</label>
                <input className="input" value={a.kind || 'practice'} onChange={(e) => setItem('activities', i, 'kind', e.target.value)} placeholder="guided_practice" />
              </div>
              <div>
                <label className="label">Recommended for state</label>
                <select className="input" value={a.target_state} onChange={(e) => setItem('activities', i, 'target_state', e.target.value)}>
                  {STATE_OPTIONS.map(([key, name]) => (
                    <option key={key} value={key}>{name}</option>
                  ))}
                </select>
              </div>
              <div className="md:col-span-2">
                <label className="label">Activity content (what the student reads or works through)</label>
                <textarea className="input min-h-[70px]" value={a.content || ''} onChange={(e) => setItem('activities', i, 'content', e.target.value)} placeholder="Imagine you are hiking down a foggy hill. The hill represents the loss landscape…" />
              </div>
              <div className="md:col-span-2">
                <label className="label">Question / short task (what the student answers to complete the activity)</label>
                <input className="input" value={a.question || ''} onChange={(e) => setItem('activities', i, 'question', e.target.value)} placeholder="What does the slope of the hill represent?" />
              </div>
              <button onClick={() => removeItem('activities', i)} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <button onClick={save} disabled={busy || !editing.name.trim()} className="btn-primary">
          {busy ? 'Saving…' : 'Save topic'}
        </button>
        <button onClick={() => { setEditing(null); setMessage(null) }} className="btn-secondary">Cancel</button>
        {editing.id && (
          <button
            onClick={() => askDelete(editing.id)}
            disabled={busy}
            aria-label={`Delete topic ${editing.name}`}
            className="ml-auto px-4 py-2 rounded-md font-semibold text-brand border border-brand/40 hover:bg-brand hover:text-white transition-colors"
          >
            Delete topic
          </button>
        )}
      </div>

      <ConfirmDeleteDialog
        noun="topic"
        preview={deleting}
        busy={busy}
        onCancel={() => setDeleting(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
