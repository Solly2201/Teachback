import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { TeacherContextBar, useTeacherContext } from '../components/TeacherContext.jsx'
import { api } from '../services/api.js'

/* Quick Lecture workflow: upload/paste material -> automatic first draft ->
   quick faculty review -> Start TeachBack (publishes the topic students see). */

const emptyForm = { title: '', description: '', objectives: '', material_text: '' }

/* The faculty workflow in three steps, shown on the lecture pages. */
function WorkflowSteps({ active }) {
  const steps = ['1. Add material', '2. Review TeachBack understanding', '3. Publish']
  return (
    <div className="card px-4 py-3 flex flex-wrap items-center gap-2 text-xs font-semibold">
      {steps.map((s, i) => (
        <span key={s} className="flex items-center gap-2">
          <span className={`px-2.5 py-1 rounded ${i === active ? 'bg-brand text-white' : 'bg-zinc-100 text-charcoal-light'}`}>{s}</span>
          {i < steps.length - 1 && <span className="text-zinc-300">→</span>}
        </span>
      ))}
      <span className="ml-auto text-charcoal-light font-normal hidden md:block">
        You review — TeachBack drafts.
      </span>
    </div>
  )
}

/* Recommended note format + optional external-AI preparation prompt.
   TeachBack itself never calls an AI service: the prompt is only text the
   teacher can copy into a tool of their own choosing. */
function NoteHelp() {
  const [help, setHelp] = useState(null)
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    api.prepPrompt().then(setHelp).catch(() => {})
  }, [])
  if (!help) return null
  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(help.prompt)
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    } catch { /* clipboard unavailable */ }
  }
  return (
    <div className="card p-4 space-y-3">
      <details>
        <summary className="text-sm font-semibold text-charcoal cursor-pointer">
          Recommended lecture note format <span className="font-normal text-charcoal-light">(optional — ordinary notes work too)</span>
        </summary>
        <pre className="mt-2 text-xs bg-zinc-50 border border-zinc-200 rounded-md p-3 overflow-x-auto whitespace-pre-wrap text-charcoal-light">{help.template}</pre>
        <p className="text-xs text-charcoal-light mt-1.5">
          Headings become concepts, &quot;Example:&quot; lines stay attached as examples, and the
          Important Connections / Common Mistakes sections are picked up directly. This format is
          recommended, not required.
        </p>
      </details>
      <div className="flex items-center gap-3 pt-2 border-t border-zinc-100">
        <button onClick={copyPrompt} className="btn-secondary text-xs">
          {copied ? '✓ Copied' : 'Copy AI preparation prompt'}
        </button>
        <span className="text-xs text-charcoal-light">
          Optional: paste this prompt plus your rough notes into an external AI assistant
          (ChatGPT, Claude, …) to convert them into the recommended format, then paste the result
          here. TeachBack itself never calls an AI service.
        </span>
      </div>
    </div>
  )
}

export default function Lectures() {
  const context = useTeacherContext()
  const { subject } = context
  const [lectures, setLectures] = useState(null)
  const [view, setView] = useState('list') // list | create | review
  const [form, setForm] = useState(emptyForm)
  const [lecture, setLecture] = useState(null) // the lecture being reviewed
  const [published, setPublished] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)

  const load = () => {
    if (!subject) return
    api.lectures(subject.id).then(setLectures).catch((e) => setMessage({ kind: 'error', text: e.message }))
  }
  useEffect(() => { load() }, [subject?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = async (file) => {
    if (!file) return
    setBusy(true)
    setMessage(null)
    try {
      if (file.name.toLowerCase().endsWith('.txt') || file.name.toLowerCase().endsWith('.md')) {
        const text = await file.text()
        setForm((f) => ({ ...f, material_text: text }))
      } else {
        const buf = await file.arrayBuffer()
        let binary = ''
        new Uint8Array(buf).forEach((b) => { binary += String.fromCharCode(b) })
        const r = await api.extractMaterial(file.name, btoa(binary))
        setForm((f) => ({ ...f, material_text: r.text }))
      }
      setMessage({ kind: 'ok', text: `Loaded material from ${file.name}.` })
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  const analyze = async () => {
    setBusy(true)
    setMessage(null)
    try {
      const lec = await api.createLecture({
        subject_id: subject.id,
        title: form.title,
        description: form.description,
        material_text: form.material_text,
        objectives: form.objectives.split('\n').map((o) => o.trim()).filter(Boolean),
      })
      setLecture(lec)
      setPublished(null)
      setView('review')
      load()
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  const openLecture = async (id) => {
    setMessage(null)
    const lec = await api.lecture(id)
    setLecture(lec)
    setPublished(null)
    setView('review')
  }

  const setDraftList = (key, updater) =>
    setLecture((l) => ({ ...l, draft: { ...l.draft, [key]: updater(l.draft[key] || []) } }))

  const saveDraft = async (silent = false) => {
    const lec = await api.updateLecture(lecture.id, {
      title: lecture.title,
      description: lecture.description,
      objectives: lecture.objectives,
      concepts: lecture.draft.concepts || [],
      relationships: lecture.draft.relationships || [],
      misconceptions: lecture.draft.misconceptions || [],
      activities: lecture.draft.activities || [],
      quiz: lecture.draft.quiz || [],
    })
    setLecture(lec)
    if (!silent) setMessage({ kind: 'ok', text: 'Draft saved.' })
    return lec
  }

  const startTeachBack = async () => {
    setBusy(true)
    setMessage(null)
    try {
      await saveDraft(true)
      const r = await api.publishLecture(lecture.id)
      setLecture(r.lecture)
      setPublished(r.topic)
      load()
    } catch (e) {
      setMessage({ kind: 'error', text: e.message })
    } finally {
      setBusy(false)
    }
  }

  const messageBox = message && (
    <div className={`card p-4 text-sm ${message.kind === 'ok' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-red-700 bg-red-50 border-red-200'}`}>
      {message.text}
    </div>
  )

  /* ---------- list ---------- */
  if (view === 'list') {
    return (
      <div className="space-y-5">
        <div className="banner">Lecture TeachBacks</div>
        <TeacherContextBar context={context} />
        <WorkflowSteps active={0} />
        {messageBox}
        <div className="flex items-center justify-between">
          <p className="text-sm text-charcoal-light max-w-xl">
            Upload today&apos;s lecture material and TeachBack prepares a first draft of the important
            concepts for you to review — no manual data entry after every class.
          </p>
          <button onClick={() => { setForm(emptyForm); setMessage(null); setView('create') }} className="btn-primary whitespace-nowrap">
            + Create Lecture TeachBack
          </button>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {!lectures && <div className="text-sm text-charcoal-light">Loading lectures…</div>}
          {lectures?.length === 0 && (
            <div className="card p-5 text-sm text-charcoal-light">No lectures yet for {subject?.name}. Create the first one.</div>
          )}
          {lectures?.map((l) => (
            <button key={l.id} onClick={() => openLecture(l.id)} className="card p-5 text-left hover:border-brand hover:shadow-md transition-all group">
              <div className="flex items-center justify-between gap-3">
                <div className="font-bold text-charcoal group-hover:text-brand">{l.title}</div>
                <span className={`text-[11px] font-bold uppercase tracking-wide px-2 py-0.5 rounded border ${
                  l.status === 'published' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-amber-50 text-amber-700 border-amber-200'
                }`}>
                  {l.status === 'published' ? 'live' : 'draft'}
                </span>
              </div>
              <p className="text-sm text-charcoal-light mt-1 line-clamp-2">{l.description}</p>
              <div className="text-xs text-charcoal-light mt-3">
                {(l.draft?.concepts || []).length} concepts · {(l.objectives || []).length} objectives
              </div>
            </button>
          ))}
        </div>
      </div>
    )
  }

  /* ---------- create ---------- */
  if (view === 'create') {
    return (
      <div className="space-y-5">
        <div className="banner">New Lecture TeachBack — {subject?.name}</div>
        <WorkflowSteps active={0} />
        {messageBox}
        <NoteHelp />
        <div className="card p-5 space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="label">Lecture title</label>
              <input className="input" value={form.title} onChange={(e) => setForm((f) => ({ ...f, title: e.target.value }))} placeholder="Backpropagation" />
            </div>
            <div>
              <label className="label">Short description (optional)</label>
              <input className="input" value={form.description} onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))} placeholder="How networks propagate error backward…" />
            </div>
          </div>
          <div>
            <label className="label">Learning objectives (optional, one per line — these get priority over automatic extraction)</label>
            <textarea className="input min-h-[70px]" value={form.objectives} onChange={(e) => setForm((f) => ({ ...f, objectives: e.target.value }))} placeholder={'Explain what backpropagation does.\nExplain what a gradient represents.'} />
          </div>
          <div>
            <label className="label">Lecture material — paste your notes, or upload a file</label>
            <textarea className="input min-h-[180px]" value={form.material_text} onChange={(e) => setForm((f) => ({ ...f, material_text: e.target.value }))} placeholder="Paste the lecture notes / slide text here…" />
            <div className="flex items-center gap-3 mt-2">
              <label className="btn-secondary cursor-pointer">
                Upload .txt / .md / .pdf
                <input type="file" accept=".txt,.md,.pdf" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
              </label>
              <span className="text-xs text-charcoal-light">The text is extracted and shown above so you can check it.</span>
            </div>
          </div>
        </div>
        <div className="flex gap-3">
          <button onClick={analyze} disabled={busy || !form.title.trim() || !form.material_text.trim()} className="btn-primary">
            {busy ? 'Analyzing material…' : 'Analyze material →'}
          </button>
          <button onClick={() => setView('list')} className="btn-secondary">Cancel</button>
        </div>
      </div>
    )
  }

  /* ---------- review ---------- */
  const draft = lecture.draft || {}
  const suggestions = lecture.suggestions || {}
  const pendingMiscons = (suggestions.misconception_suggestions || []).filter(
    (s) => !(draft.misconceptions || []).some((m) => m.name === s.name)
  )

  if (published) {
    return (
      <div className="space-y-5">
        <div className="banner">TeachBack is live — {lecture.title}</div>
        <div className="card p-8 text-center">
          <div className="text-3xl text-emerald-600 mb-2">✓</div>
          <div className="text-lg font-bold text-charcoal">Students can now TeachBack this lecture</div>
          <p className="text-sm text-charcoal-light mt-1.5 max-w-lg mx-auto">
            &quot;{published.name}&quot; is available in the student TeachBack list with {published.concepts.length} concepts.
            You can refine it any time here or in Topic Management.
          </p>
          <div className="flex justify-center gap-3 mt-6">
            <button onClick={() => { setPublished(null) }} className="btn-secondary">Keep editing</button>
            <button onClick={() => setView('list')} className="btn-primary">Back to lectures</button>
            <Link to="/topics" className="btn-secondary">Open Topic Management</Link>
          </div>
        </div>
      </div>
    )
  }

  const setConcept = (i, patch) =>
    setDraftList('concepts', (l) => l.map((x, j) => (j === i ? { ...x, ...patch } : x)))

  return (
    <div className="space-y-5">
      <div className="banner">TeachBack Preview — {lecture.title}</div>
      <WorkflowSteps active={1} />
      {messageBox}
      <p className="text-sm text-charcoal-light">
        This is the automatic first draft from your lecture material. Rename, remove or add anything —
        nothing is final until you press <strong>Start TeachBack</strong>. The meaning of each concept
        should be what a student should be able to say back in their own words, not a textbook definition.
      </p>

      {/* concepts */}
      <div className="card">
        <div className="card-header">
          <span>{(draft.concepts || []).length} concepts detected</span>
          <button onClick={() => setDraftList('concepts', (l) => [...l, { name: '', description: '', facts: [], examples: [] }])} className="text-white/90 hover:text-white normal-case font-normal">+ add concept</button>
        </div>
        <div className="p-4 space-y-3">
          {(draft.concepts || []).map((c, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 space-y-3 relative">
              <div className="grid md:grid-cols-3 gap-3">
                <div>
                  <label className="label">Concept</label>
                  <input className="input" value={c.name} onChange={(e) => setConcept(i, { name: e.target.value })} />
                </div>
                <div className="md:col-span-2">
                  <label className="label">Meaning — what a student should be able to say back</label>
                  <input className="input" value={c.description} onChange={(e) => setConcept(i, { description: e.target.value })} />
                </div>
              </div>
              <div>
                <label className="label">Important facts (one per line — students get credit for expressing these)</label>
                <textarea
                  className="input min-h-[48px] text-sm"
                  value={(c.facts || []).join('\n')}
                  onChange={(e) => setConcept(i, { facts: e.target.value.split('\n') })}
                  onBlur={(e) => setConcept(i, { facts: e.target.value.split('\n').map((f) => f.trim()).filter(Boolean) })}
                />
              </div>
              {(c.source_section || (c.source_sentences || []).length > 0 || (c.examples || []).length > 0) && (
                <div className="text-xs text-charcoal-light bg-zinc-50 border border-zinc-100 rounded p-2">
                  <span className="font-semibold text-charcoal">Why suggested? </span>
                  {c.source_section && <>Found as the lecture section &quot;{c.source_section}&quot;. </>}
                  {(c.source_sentences || [])[0] && <>Supported by: &quot;{c.source_sentences[0]}&quot; </>}
                  {(c.examples || []).length > 0 && <>· Example kept: <code>{c.examples[0]}</code></>}
                </div>
              )}
              <details>
                <summary className="text-xs font-semibold text-charcoal-light cursor-pointer uppercase tracking-wide">Questions (drafted — edit if you like)</summary>
                <div className="grid md:grid-cols-2 gap-3 mt-2">
                  {[['main_question', 'Main question'], ['easier_question', 'Easier question'],
                    ['probe_question', 'Probe'], ['application_question', 'Application (optional extension — never required)']].map(([key, label]) => (
                    <div key={key}>
                      <label className="label">{label}</label>
                      <input className="input" value={c[key] || ''} onChange={(e) => setConcept(i, { [key]: e.target.value })} />
                    </div>
                  ))}
                </div>
              </details>
              <button onClick={() => setDraftList('concepts', (l) => l.filter((_, j) => j !== i))} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
          {(suggestions.concepts || []).length > 0 && (
            <p className="text-xs text-charcoal-light">
              Originally suggested from the material: {(suggestions.concepts || []).map((c) => c.name).join(', ')}.
            </p>
          )}
        </div>
      </div>

      {/* relationships */}
      <div className="card">
        <div className="card-header">
          <span>Potential relationships</span>
          <button onClick={() => setDraftList('relationships', (l) => [...l, { source: '', label: 'relates to', target: '', description: '' }])} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          {(draft.relationships || []).length === 0 && <p className="text-sm text-charcoal-light">No relationships kept — optional.</p>}
          {(draft.relationships || []).map((r, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-3 gap-3 relative">
              <div>
                <label className="label">Source</label>
                <input className="input" value={r.source} onChange={(e) => setDraftList('relationships', (l) => l.map((x, j) => (j === i ? { ...x, source: e.target.value } : x)))} />
              </div>
              <div>
                <label className="label">Link</label>
                <input className="input" value={r.label} onChange={(e) => setDraftList('relationships', (l) => l.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))} />
              </div>
              <div>
                <label className="label">Target</label>
                <input className="input" value={r.target} onChange={(e) => setDraftList('relationships', (l) => l.map((x, j) => (j === i ? { ...x, target: e.target.value } : x)))} />
              </div>
              <div className="md:col-span-3">
                <label className="label">Correct sentence</label>
                <input className="input" value={r.description} onChange={(e) => setDraftList('relationships', (l) => l.map((x, j) => (j === i ? { ...x, description: e.target.value } : x)))} />
              </div>
              <button onClick={() => setDraftList('relationships', (l) => l.filter((_, j) => j !== i))} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      {/* objectives */}
      <div className="card">
        <div className="card-header">Learning objectives</div>
        <div className="p-4">
          <textarea
            className="input min-h-[80px]"
            value={(lecture.objectives || []).join('\n')}
            onChange={(e) => setLecture((l) => ({ ...l, objectives: e.target.value.split('\n') }))}
          />
          <p className="text-xs text-charcoal-light mt-1.5">One objective per line. Your own objectives always take priority over the automatic draft.</p>
        </div>
      </div>

      {/* misconceptions */}
      <div className="card">
        <div className="card-header">
          <span>Misconceptions (teacher-reviewed)</span>
          <button onClick={() => setDraftList('misconceptions', (l) => [...l, { name: '', description: '', clarification: '', probe_question: '' }])} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          {pendingMiscons.map((m) => (
            <div key={m.name} className="border border-amber-200 bg-amber-50 rounded-md p-3 flex items-start justify-between gap-3">
              <div className="text-sm">
                <div className="font-semibold text-charcoal">Suggested: {m.name}</div>
                <div className="text-charcoal-light text-xs mt-0.5">{m.description}</div>
              </div>
              <button onClick={() => setDraftList('misconceptions', (l) => [...l, { name: m.name, description: m.description, clarification: m.clarification, probe_question: m.probe_question }])} className="btn-secondary whitespace-nowrap">
                Accept
              </button>
            </div>
          ))}
          {(draft.misconceptions || []).map((m, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-2 gap-3 relative">
              <div>
                <label className="label">Name</label>
                <input className="input" value={m.name} onChange={(e) => setDraftList('misconceptions', (l) => l.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} />
              </div>
              <div>
                <label className="label">Wrong claim</label>
                <input className="input" value={m.description} onChange={(e) => setDraftList('misconceptions', (l) => l.map((x, j) => (j === i ? { ...x, description: e.target.value } : x)))} />
              </div>
              <div>
                <label className="label">Correct clarification</label>
                <input className="input" value={m.clarification} onChange={(e) => setDraftList('misconceptions', (l) => l.map((x, j) => (j === i ? { ...x, clarification: e.target.value } : x)))} />
              </div>
              <div>
                <label className="label">Probe question</label>
                <input className="input" value={m.probe_question} onChange={(e) => setDraftList('misconceptions', (l) => l.map((x, j) => (j === i ? { ...x, probe_question: e.target.value } : x)))} />
              </div>
              <button onClick={() => setDraftList('misconceptions', (l) => l.filter((_, j) => j !== i))} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
          {(draft.misconceptions || []).length === 0 && pendingMiscons.length === 0 && (
            <p className="text-sm text-charcoal-light">None — misconceptions stay teacher-authored; add one if you expect a common mix-up.</p>
          )}
        </div>
      </div>

      {/* knowledge check (10 MCQs, teacher-reviewed) */}
      <div className="card">
        <div className="card-header">
          <span>Knowledge check ({(draft.quiz || []).length} questions — secondary evidence, TeachBack stays primary)</span>
          <span className="flex gap-3">
            <button
              onClick={async () => { setLecture(await api.regenerateQuiz(lecture.id, null)) }}
              className="text-white/90 hover:text-white normal-case font-normal"
            >
              ↻ regenerate all
            </button>
            <button
              onClick={() => setDraftList('quiz', (l) => [...l, { concept_name: '', kind: 'basic', question: '', options: ['', '', '', ''], correct_index: 0, explanation: '' }])}
              className="text-white/90 hover:text-white normal-case font-normal"
            >
              + add
            </button>
          </span>
        </div>
        <div className="p-4 space-y-3">
          {(draft.quiz || []).length === 0 && (
            <p className="text-sm text-charcoal-light">No questions — the knowledge check is optional; add or regenerate if you want one.</p>
          )}
          {(draft.quiz || []).map((q, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 space-y-2 relative">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-bold text-charcoal">Q{i + 1}</span>
                <span className="px-1.5 py-0.5 rounded bg-zinc-100 text-charcoal-light uppercase tracking-wide font-semibold">{q.kind}</span>
                <input
                  className="input py-1 w-40 text-xs"
                  placeholder="Concept"
                  value={q.concept_name}
                  onChange={(e) => setDraftList('quiz', (l) => l.map((x, j) => (j === i ? { ...x, concept_name: e.target.value } : x)))}
                />
                <button
                  onClick={async () => {
                    try { setLecture(await api.regenerateQuiz(lecture.id, i)) }
                    catch (e) { setMessage({ kind: 'error', text: e.message }) }
                  }}
                  className="ml-auto text-charcoal-light hover:text-brand font-semibold"
                >
                  ↻ regenerate
                </button>
              </div>
              <textarea
                className="input min-h-[40px] text-sm"
                value={q.question}
                onChange={(e) => setDraftList('quiz', (l) => l.map((x, j) => (j === i ? { ...x, question: e.target.value } : x)))}
              />
              <div className="grid md:grid-cols-2 gap-2">
                {(q.options || []).map((opt, oi) => (
                  <div key={oi} className="flex items-center gap-2">
                    <input
                      type="radio"
                      name={`correct-${i}`}
                      checked={q.correct_index === oi}
                      onChange={() => setDraftList('quiz', (l) => l.map((x, j) => (j === i ? { ...x, correct_index: oi } : x)))}
                      className="accent-[#A5231B] shrink-0"
                      title="Mark as the correct answer"
                    />
                    <input
                      className={`input py-1.5 text-sm ${q.correct_index === oi ? 'border-emerald-400' : ''}`}
                      value={opt}
                      onChange={(e) => setDraftList('quiz', (l) => l.map((x, j) => (j === i ? { ...x, options: x.options.map((o, k) => (k === oi ? e.target.value : o)) } : x)))}
                    />
                  </div>
                ))}
              </div>
              <div>
                <label className="label">Explanation shown after answering</label>
                <input
                  className="input text-sm"
                  value={q.explanation}
                  onChange={(e) => setDraftList('quiz', (l) => l.map((x, j) => (j === i ? { ...x, explanation: e.target.value } : x)))}
                />
              </div>
              <button onClick={() => setDraftList('quiz', (l) => l.filter((_, j) => j !== i))} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      {/* suggested activities */}
      <div className="card">
        <div className="card-header">
          <span>Suggested activities (from this lecture&apos;s own concepts)</span>
          <button onClick={() => setDraftList('activities', (l) => [...l, { target_state: 'understanding', kind: 'practice', title: '', description: '', content: '', question: '' }])} className="text-white/90 hover:text-white normal-case font-normal">+ add</button>
        </div>
        <div className="p-4 space-y-3">
          {(draft.activities || []).length === 0 && (
            <p className="text-sm text-charcoal-light">No activities — a generic fallback will be used per learning state.</p>
          )}
          {(draft.activities || []).map((a, i) => (
            <div key={i} className="border border-zinc-200 rounded-md p-3 grid md:grid-cols-4 gap-3 relative">
              <div>
                <label className="label">For students who are…</label>
                <select className="input" value={a.target_state} onChange={(e) => setDraftList('activities', (l) => l.map((x, j) => (j === i ? { ...x, target_state: e.target.value } : x)))}>
                  <option value="not_trying">Not engaging</option>
                  <option value="unclear">Unclear</option>
                  <option value="struggling">Struggling</option>
                  <option value="understanding">Understanding</option>
                  <option value="confident">Confident (extension)</option>
                </select>
              </div>
              <div className="md:col-span-3">
                <label className="label">Title</label>
                <input className="input" value={a.title} onChange={(e) => setDraftList('activities', (l) => l.map((x, j) => (j === i ? { ...x, title: e.target.value } : x)))} />
              </div>
              <div className="md:col-span-2">
                <label className="label">Material the student reads</label>
                <textarea className="input min-h-[48px] text-sm" value={a.content} onChange={(e) => setDraftList('activities', (l) => l.map((x, j) => (j === i ? { ...x, content: e.target.value } : x)))} />
              </div>
              <div className="md:col-span-2">
                <label className="label">Task / question</label>
                <textarea className="input min-h-[48px] text-sm" value={a.question} onChange={(e) => setDraftList('activities', (l) => l.map((x, j) => (j === i ? { ...x, question: e.target.value } : x)))} />
              </div>
              <button onClick={() => setDraftList('activities', (l) => l.filter((_, j) => j !== i))} className="absolute -top-2 -right-2 w-6 h-6 bg-white border border-zinc-300 rounded-full text-charcoal-light hover:text-brand hover:border-brand text-xs">✕</button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-3">
        <button onClick={startTeachBack} disabled={busy || !(draft.concepts || []).some((c) => c.name.trim())} className="btn-primary">
          {busy ? 'Publishing…' : lecture.status === 'published' ? 'Update TeachBack' : 'Start TeachBack'}
        </button>
        <button onClick={() => saveDraft()} disabled={busy} className="btn-secondary">Save draft</button>
        <button onClick={() => { setView('list'); setMessage(null) }} className="btn-secondary">Back</button>
      </div>
    </div>
  )
}
