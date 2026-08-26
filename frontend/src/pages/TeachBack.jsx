import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import Meter from '../components/Meter.jsx'
import StateBadge from '../components/StateBadge.jsx'
import { api, STATE_DESCRIPTIONS } from '../services/api.js'

/* Visual states for the concept progress timeline. */
const TIMELINE_META = {
  done: { icon: '✓', cls: 'bg-emerald-50 border-emerald-300 text-emerald-800' },
  clarify: { icon: '⚠', cls: 'bg-amber-50 border-amber-300 text-amber-800' },
  current: { icon: '●', cls: 'bg-brand text-white border-brand' },
  pending: { icon: '○', cls: 'bg-white border-zinc-300 text-charcoal-light' },
}

function ConceptTimeline({ timeline, conceptNo, totalConcepts, questionNo }) {
  if (!timeline?.length) return null
  return (
    <div className="card p-4">
      <div className="flex flex-wrap items-center gap-y-2">
        {timeline.map((c, i) => {
          const meta = TIMELINE_META[c.status] || TIMELINE_META.pending
          return (
            <div key={c.name} className="flex items-center">
              <span
                className={`inline-flex items-center gap-1.5 border rounded px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide ${meta.cls}`}
                title={c.name}
              >
                <span aria-hidden="true">{meta.icon}</span>
                {c.name}
              </span>
              {i < timeline.length - 1 && <span className="mx-1.5 text-zinc-300">→</span>}
            </div>
          )
        })}
      </div>
      <div className="mt-2.5 pt-2.5 border-t border-zinc-100 flex items-center justify-between text-xs text-charcoal-light">
        <span>
          Concept <strong className="text-charcoal">{conceptNo}</strong> of {totalConcepts}
          <span className="mx-2 text-zinc-300">·</span>
          Question {questionNo}
        </span>
        <span className="hidden sm:flex items-center gap-3">
          <span><span className="text-emerald-600">✓</span> understood</span>
          <span><span className="text-amber-600">⚠</span> to clarify</span>
          <span><span className="text-brand">●</span> current</span>
        </span>
      </div>
    </div>
  )
}

export default function TeachBack({ user }) {
  const [topics, setTopics] = useState(null)
  const [session, setSession] = useState(null)
  const [progress, setProgress] = useState(null) // timeline / concept_no / question_no
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [misconceptions, setMisconceptions] = useState([]) // {name, resolved}
  const [awaitingReport, setAwaitingReport] = useState(false)
  const [report, setReport] = useState({ attention: 7, confidence: 5, difficulty: 5 })
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    api.topics().then(setTopics).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, awaitingReport, result])

  const start = async (topicId) => {
    setBusy(true)
    setError(null)
    try {
      const s = await api.startSession(user.id, topicId)
      setSession(s)
      setProgress({
        timeline: s.timeline, concept_no: s.concept_no,
        total_concepts: s.total_concepts, question_no: s.question_no,
      })
      setMessages([
        { who: 'system', text: s.intro, intro: true },
        { who: 'system', text: s.prompt },
      ])
      setMisconceptions([])
      setResult(null)
      setAwaitingReport(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const send = async () => {
    if (!text.trim() || busy) return
    const studentText = text.trim()
    setBusy(true)
    setError(null)
    setMessages((m) => [...m, { who: 'student', text: studentText }])
    setText('')
    try {
      const r = await api.respond(session.session_id, studentText)
      setProgress({
        timeline: r.timeline, concept_no: r.concept_no,
        total_concepts: r.total_concepts, question_no: r.question_no,
      })
      if (r.misconception) {
        setMisconceptions((ms) => [...ms.filter((x) => x.name !== r.misconception.name),
          { name: r.misconception.name, resolved: false }])
      }
      if (r.resolved_misconception) {
        setMisconceptions((ms) => ms.map((x) =>
          x.name === r.resolved_misconception ? { ...x, resolved: true } : x))
      }
      const sysMsg = {
        who: 'system',
        feedback: r.feedback,
        resolved: r.resolved_misconception || null,
        incidental: r.incidental?.length ? r.incidental : null,
        text: r.followup ? r.followup.text : r.closing,
        reason: r.followup?.reason,
      }
      setMessages((m) => [...m, sysMsg])
      if (!r.followup) setAwaitingReport(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const finish = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await api.finish(session.session_id, report)
      setResult(r)
      setAwaitingReport(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const reset = () => {
    setSession(null)
    setProgress(null)
    setMessages([])
    setMisconceptions([])
    setResult(null)
    setAwaitingReport(false)
  }

  /* ---------- topic selection ---------- */
  if (!session) {
    return (
      <div className="space-y-5">
        <div className="banner">TeachBack — Choose a Topic</div>
        {error && <div className="card p-4 text-sm text-red-700 bg-red-50 border-red-200">{error}</div>}
        <p className="text-sm text-charcoal-light">
          Pick a topic you have studied. We&apos;ll have a <strong>short conversation</strong> about it —
          simple questions, one at a time. Most answers only need <strong>one or two sentences</strong>;
          the system adapts its follow-up questions to what you say.
        </p>
        <div className="grid md:grid-cols-3 gap-4">
          {!topics && <div className="text-charcoal-light text-sm">Loading topics…</div>}
          {topics?.map((t) => (
            <button
              key={t.id}
              onClick={() => start(t.id)}
              disabled={busy}
              className="card p-5 text-left hover:border-brand hover:shadow-md transition-all group"
            >
              <div className="font-bold text-charcoal group-hover:text-brand">{t.name}</div>
              <p className="text-sm text-charcoal-light mt-1.5 line-clamp-2">{t.description}</p>
              <div className="text-xs text-charcoal-light mt-3">
                {t.concept_count} concepts · about {t.concept_count}–{t.concept_count * 2} short questions
              </div>
            </button>
          ))}
        </div>
      </div>
    )
  }

  /* ---------- session result ---------- */
  if (result) {
    const demonstrated = result.concept_summary.filter((c) => c.status === 'covered')
    const needsWork = result.concept_summary.filter((c) => c.status !== 'covered')
    const relsDemonstrated = (result.relationship_summary || []).filter((r) => r.status === 'demonstrated')
    const relsUnclear = (result.relationship_summary || []).filter((r) => r.status === 'needs_clarification')
    return (
      <div className="space-y-5">
        <div className="banner">Session Complete — {session.topic.name}</div>

        <div className="card">
          <div className="card-header">Current learning state</div>
          <div className="p-5">
            <div className="flex flex-col sm:flex-row sm:items-center gap-4">
              <StateBadge label={result.state.label} size="lg" />
              <p className="text-sm text-charcoal-light">{STATE_DESCRIPTIONS[result.state.label]}</p>
            </div>
            {result.previous_state_label && result.previous_state_label !== result.state.label && (
              <div className="mt-4 flex items-center gap-3 text-sm">
                <span className="text-xs font-semibold text-charcoal-light uppercase tracking-wide">Learning journey</span>
                <StateBadge label={result.previous_state_label} />
                <span className="text-zinc-400">→</span>
                <StateBadge label={result.state.label} />
              </div>
            )}
            <div className="mt-4 pt-4 border-t border-zinc-100">
              <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-2">
                How sure is the model? (HMM posterior)
              </div>
              {Object.entries(result.state.posterior).map(([name, p]) => (
                <Meter key={name} label={name} value={p} color={name === result.state.label ? '#A5231B' : '#A1A1AA'} />
              ))}
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-5">
          <div className="card">
            <div className="card-header">You demonstrated</div>
            <ul className="p-4 space-y-2">
              {demonstrated.map((c) => (
                <li key={c.name} className="flex items-center gap-2 text-sm">
                  <span className="text-emerald-600">✓</span>
                  <span className="text-charcoal">{c.name}</span>
                </li>
              ))}
              {demonstrated.length === 0 && (
                <li className="text-sm text-charcoal-light">No concepts were clearly demonstrated this time — that&apos;s okay, it tells us where to focus.</li>
              )}
              {relsDemonstrated.length > 0 && (
                <li className="pt-2 mt-1 border-t border-zinc-100">
                  <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1.5">Connections you made</div>
                  <ul className="space-y-1.5">
                    {relsDemonstrated.map((r) => (
                      <li key={`${r.source}-${r.target}`} className="flex items-center gap-2 text-sm">
                        <span className="text-emerald-600">✓</span>
                        <span className="text-charcoal">{r.source} <span className="text-zinc-400">→</span> {r.target}</span>
                      </li>
                    ))}
                  </ul>
                </li>
              )}
              {result.resolved_misconceptions?.map((m) => (
                <li key={m} className="flex items-start gap-2 text-sm">
                  <span className="text-emerald-600">✓</span>
                  <span className="text-charcoal">Cleared up: <em>{m}</em></span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card">
            <div className="card-header">Needs clarification</div>
            <ul className="p-4 space-y-2">
              {needsWork.map((c) => (
                <li key={c.name} className="flex items-center gap-2 text-sm">
                  <span className={c.status === 'missing' ? 'text-zinc-400' : 'text-amber-600'}>
                    {c.status === 'missing' ? '○' : '⚠'}
                  </span>
                  <span className="text-charcoal">{c.name}</span>
                  {c.status === 'partial' && <span className="text-xs text-charcoal-light">(partly there)</span>}
                </li>
              ))}
              {relsUnclear.map((r) => (
                <li key={`${r.source}-${r.target}`} className="flex items-center gap-2 text-sm">
                  <span className="text-amber-600">⚠</span>
                  <span className="text-charcoal">{r.source} <span className="text-zinc-400">→</span> {r.target}</span>
                  <span className="text-xs text-charcoal-light">(connection)</span>
                </li>
              ))}
              {needsWork.length === 0 && relsUnclear.length === 0 && (
                <li className="text-sm text-emerald-700">Nothing — every concept was demonstrated. Well done.</li>
              )}
            </ul>
          </div>
        </div>

        {result.misconception_details?.length > 0 && (
          <div className="card">
            <div className="card-header">Misconceptions we talked about</div>
            <div className="p-4 space-y-3">
              {result.misconception_details.map((m) => (
                <div key={m.name} className={`border rounded-md p-3 ${m.resolved ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-charcoal">{m.name}</div>
                    <span className={`text-[11px] font-bold uppercase tracking-wide ${m.resolved ? 'text-emerald-700' : 'text-amber-700'}`}>
                      {m.resolved ? '✓ resolved' : 'still open'}
                    </span>
                  </div>
                  <p className="text-sm text-charcoal-light mt-1">{m.clarification}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="card">
          <div className="card-header">Recommended next activity</div>
          <div className="p-5">
            <div className="text-xs font-bold uppercase tracking-wide text-brand mb-1">
              {result.recommendation.activity.kind.replace('_', ' ')}
            </div>
            <div className="font-semibold text-charcoal">{result.recommendation.activity.title}</div>
            <p className="text-sm text-charcoal-light mt-1.5">{result.recommendation.activity.description}</p>
            {result.recommendation.why && (
              <div className="mt-3 pt-3 border-t border-zinc-100">
                <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1">Why this was recommended</div>
                <p className="text-sm text-charcoal">{result.recommendation.why}</p>
              </div>
            )}
            {result.recommendation.notes?.map((n, i) => (
              <div key={i} className="mt-3 text-xs bg-amber-50 border border-amber-200 text-amber-800 rounded-md p-2">{n}</div>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          <button onClick={reset} className="btn-primary">Start next activity — new topic</button>
          <Link to="/progress" className="btn-secondary">View my progress</Link>
        </div>
      </div>
    )
  }

  /* ---------- guided conversation ---------- */
  return (
    <div className="space-y-4">
      <div className="banner">TeachBack — {session.topic.name}</div>
      {error && <div className="card p-4 text-sm text-red-700 bg-red-50 border-red-200">{error}</div>}

      {progress && (
        <ConceptTimeline
          timeline={progress.timeline}
          conceptNo={progress.concept_no}
          totalConcepts={progress.total_concepts}
          questionNo={progress.question_no}
        />
      )}

      <div className="grid lg:grid-cols-3 gap-5 items-start">
        <div className="lg:col-span-2 card flex flex-col">
          <div className="card-header">
            <span>Conversation</span>
            <span className="normal-case font-normal text-white/80">
              Concept {progress?.concept_no} of {progress?.total_concepts}
            </span>
          </div>
          <div className="p-4 space-y-3 min-h-[300px] max-h-[440px] overflow-y-auto">
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.who === 'student' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[85%] rounded-lg px-4 py-2.5 text-sm ${
                    m.who === 'student'
                      ? 'bg-brand text-white rounded-br-none'
                      : m.intro
                        ? 'bg-zinc-50 border border-zinc-200 text-charcoal-light rounded-bl-none'
                        : 'bg-zinc-100 text-charcoal rounded-bl-none'
                  }`}
                >
                  {m.feedback && <div className="font-semibold mb-1">{m.feedback}</div>}
                  {m.resolved && (
                    <div className="text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-2 py-1 mb-1.5">
                      ✓ Misconception resolved: {m.resolved}
                    </div>
                  )}
                  {m.incidental && (
                    <div className="text-xs text-charcoal-light mb-1.5">
                      You already touched on {m.incidental.join(', ')} — we can skip that.
                    </div>
                  )}
                  {m.text}
                </div>
              </div>
            ))}
            {busy && <div className="text-xs text-charcoal-light italic px-2">Thinking about your answer…</div>}
            <div ref={bottomRef} />
          </div>

          {awaitingReport ? (
            <div className="border-t border-zinc-200 p-4 space-y-4">
              <div className="text-sm font-semibold text-charcoal">Last step — how did this session feel?</div>
              {[
                ['attention', 'How focused were you this session?'],
                ['confidence', 'How confident do you feel about this topic now?'],
                ['difficulty', 'How difficult did this topic feel?'],
              ].map(([key, label]) => (
                <div key={key}>
                  <div className="flex justify-between text-xs text-charcoal-light mb-1">
                    <span>{label}</span>
                    <span className="font-bold text-charcoal tabular-nums">{report[key]}/10</span>
                  </div>
                  <input
                    type="range" min="0" max="10" step="1" value={report[key]}
                    onChange={(e) => setReport((r) => ({ ...r, [key]: Number(e.target.value) }))}
                    className="w-full accent-[#A5231B]"
                  />
                </div>
              ))}
              <button onClick={finish} disabled={busy} className="btn-primary w-full">
                {busy ? 'Estimating your learning state…' : 'See my session summary'}
              </button>
            </div>
          ) : (
            <div className="border-t border-zinc-200 p-4">
              <textarea
                className="input min-h-[64px] resize-y"
                placeholder="One or two sentences is enough…"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send()
                  }
                }}
                disabled={busy}
              />
              <div className="flex items-center justify-between mt-2">
                <span className="text-xs text-charcoal-light">Enter to send · Shift+Enter for a new line</span>
                <button onClick={send} disabled={busy || !text.trim()} className="btn-primary">
                  Send answer
                </button>
              </div>
            </div>
          )}
        </div>

        {/* session side panel */}
        <div className="space-y-5">
          <div className="card">
            <div className="card-header">This session</div>
            <div className="p-4 space-y-3">
              <ul className="space-y-1.5">
                {progress?.timeline.map((c) => {
                  const meta = TIMELINE_META[c.status] || TIMELINE_META.pending
                  return (
                    <li key={c.name} className="flex items-center gap-2 text-sm">
                      <span className={
                        c.status === 'done' ? 'text-emerald-600'
                          : c.status === 'clarify' ? 'text-amber-600'
                            : c.status === 'current' ? 'text-brand' : 'text-zinc-300'
                      }>
                        {meta.icon}
                      </span>
                      <span className={c.status === 'pending' ? 'text-charcoal-light' : 'text-charcoal'}>
                        {c.name}
                      </span>
                    </li>
                  )
                })}
              </ul>
              {misconceptions.length > 0 && (
                <div className="pt-3 border-t border-zinc-100 space-y-1.5">
                  <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide">Ideas we untangled</div>
                  {misconceptions.map((m) => (
                    <div key={m.name} className={`text-xs rounded border px-2 py-1.5 ${
                      m.resolved ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-amber-50 border-amber-200 text-amber-800'
                    }`}>
                      {m.resolved ? '✓ ' : '⚠ '}{m.name}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="card p-4 text-xs text-charcoal-light leading-relaxed">
            <strong className="text-charcoal">How this works:</strong> your answers are compared with the topic&apos;s
            concepts and known misconceptions using sentence embeddings. The next question is chosen by fixed rules
            from that analysis — the final learning state comes from a Hidden Markov Model over all your sessions.
            There are no wrong-answer penalties; the goal is to find out what you understand.
          </div>
        </div>
      </div>
    </div>
  )
}
