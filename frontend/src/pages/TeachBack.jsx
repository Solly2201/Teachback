import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import Meter from '../components/Meter.jsx'
import StateBadge from '../components/StateBadge.jsx'
import { api, studentStateDescription, studentStateLabel } from '../services/api.js'

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

/* The optional "Quick knowledge check": 10 short teacher-reviewed questions.
   Secondary evidence — it never replaces the student's own explanation. */
function KnowledgeCheck({ quizInfo, topicId, studentId, sessionId, onResult }) {
  const [quiz, setQuiz] = useState(null)
  const [current, setCurrent] = useState(0)
  const [answers, setAnswers] = useState({}) // question_id -> selected index
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const start = async () => {
    setBusy(true)
    try {
      const q = await api.topicQuiz(topicId)
      if (q.available) setQuiz(q)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const submit = async (finalAnswers) => {
    setBusy(true)
    try {
      const r = await api.submitQuiz(quiz.quiz_id, {
        student_id: studentId,
        session_id: sessionId,
        answers: Object.entries(finalAnswers).map(([qid, idx]) => ({
          question_id: Number(qid), selected_index: idx,
        })),
      })
      setResult(r)
      onResult?.(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (!quizInfo) return null

  if (result) {
    return (
      <div className="card">
        <div className="card-header">Knowledge check complete</div>
        <div className="p-5 space-y-4">
          <div className="text-lg font-bold text-charcoal">{result.headline}</div>
          <p className="text-sm text-charcoal-light">{result.message}</p>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1.5">Looked solid</div>
              <ul className="space-y-1">
                {result.solid_concepts.map((n) => (
                  <li key={n} className="text-sm text-charcoal flex gap-2"><span className="text-emerald-600">✓</span>{n}</li>
                ))}
                {result.solid_concepts.length === 0 && <li className="text-sm text-charcoal-light">—</li>}
              </ul>
            </div>
            <div>
              <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1.5">Worth revisiting</div>
              <ul className="space-y-1">
                {result.revisit_concepts.map((n) => (
                  <li key={n} className="text-sm text-charcoal flex gap-2"><span className="text-amber-600">⚠</span>{n}</li>
                ))}
                {result.revisit_concepts.length === 0 && <li className="text-sm text-emerald-700">Nothing — well done.</li>}
              </ul>
            </div>
          </div>
          {result.relationship_evidence?.filter((r) => r.message).length > 0 && (
            <div className="pt-3 border-t border-zinc-100">
              <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1.5">Connections</div>
              <ul className="space-y-1.5">
                {result.relationship_evidence.filter((r) => r.message).map((r) => (
                  <li key={`${r.source}-${r.target}`} className="text-xs text-charcoal-light">
                    <span className="font-semibold text-charcoal">
                      {r.source} <span className="text-zinc-400">→</span> {r.target}
                    </span>{' '}
                    <span className="text-charcoal-light">
                      (TeachBack: {r.teachback_label}
                      {r.mcq_total ? ` · knowledge check: ${r.mcq_correct}/${r.mcq_total}` : ''})
                    </span>{' '}
                    {r.message}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.combined.filter((c) => c.message).length > 0 && (
            <div className="space-y-1.5 pt-3 border-t border-zinc-100">
              {result.combined.filter((c) => c.message).map((c) => (
                <p key={c.name} className="text-xs text-charcoal-light">
                  <span className="font-semibold text-charcoal">{c.name}:</span> {c.message}
                </p>
              ))}
            </div>
          )}
          <details>
            <summary className="text-xs font-semibold text-charcoal-light uppercase tracking-wide cursor-pointer">Review the questions</summary>
            <div className="mt-2 space-y-3">
              {result.per_question.map((q, i) => (
                <div key={q.id} className={`border rounded-md p-3 text-sm ${q.correct ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
                  <div className="font-medium text-charcoal">{i + 1}. {q.question}</div>
                  <div className="text-xs mt-1 text-charcoal-light">
                    Your answer: {q.options[q.selected_index]} {q.correct ? '✓' : `· Correct: ${q.options[q.correct_index]}`}
                  </div>
                  <div className="text-xs mt-1 text-charcoal">{q.explanation}</div>
                </div>
              ))}
            </div>
          </details>
        </div>
      </div>
    )
  }

  if (!quiz) {
    return (
      <div className="card p-5">
        <div className="text-sm font-semibold text-charcoal">Quick knowledge check <span className="font-normal text-charcoal-light">(optional)</span></div>
        <p className="text-xs text-charcoal-light mt-1.5 max-w-xl">
          {quizInfo.n_questions} short questions about today&apos;s lecture. Don&apos;t worry if you don&apos;t
          remember everything — this just helps us understand what parts may need another look. It is
          not a replacement for your explanation.
        </p>
        {error && <p className="text-xs text-red-700 mt-2">{error}</p>}
        <button onClick={start} disabled={busy} className="btn-secondary mt-3">
          {busy ? 'Loading…' : 'Start knowledge check'}
        </button>
      </div>
    )
  }

  const q = quiz.questions[current]
  const selected = answers[q.id]
  return (
    <div className="card">
      <div className="card-header">
        <span>Quick knowledge check</span>
        <span className="normal-case font-normal text-white/80">Question {current + 1} of {quiz.questions.length}</span>
      </div>
      <div className="p-5 space-y-3">
        <div className="h-1.5 bg-zinc-100 rounded overflow-hidden">
          <div className="h-full bg-brand rounded-r transition-all" style={{ width: `${((current) / quiz.questions.length) * 100}%` }} />
        </div>
        <div className="text-sm font-medium text-charcoal">{q.question}</div>
        <div className="space-y-1.5">
          {q.options.map((opt, i) => (
            <button
              key={i}
              onClick={() => setAnswers((a) => ({ ...a, [q.id]: i }))}
              className={`w-full text-left text-sm border rounded-md px-3 py-2 transition-colors ${
                selected === i ? 'border-brand bg-red-50 text-charcoal' : 'border-zinc-200 hover:border-brand text-charcoal'
              }`}
            >
              <span className="font-semibold mr-2">{String.fromCharCode(65 + i)}.</span>{opt}
            </button>
          ))}
        </div>
        <div className="flex justify-between items-center pt-2">
          <button onClick={() => setCurrent((c) => Math.max(0, c - 1))} disabled={current === 0} className="btn-secondary disabled:opacity-40">← Back</button>
          {current < quiz.questions.length - 1 ? (
            <button onClick={() => setCurrent((c) => c + 1)} disabled={selected == null} className="btn-primary disabled:opacity-40">Next →</button>
          ) : (
            <button onClick={() => submit(answers)} disabled={selected == null || busy} className="btn-primary disabled:opacity-40">
              {busy ? 'Checking…' : 'Finish check'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

const PACE_OPTIONS = ['Too slow', 'A little slow', 'About right', 'A little fast', 'Too fast']
const FEEDBACK_CHIPS = ['More examples', 'Slower explanation', 'Faster pace', 'More practice',
  'More visuals', 'More real-world examples', 'More time for questions', 'Less repetition',
  'More challenging material']

/* How settled the learning-condition reading is, in words rather than numbers.
   Students get the confidence, not the classifier. */
function settledness(posterior) {
  const values = Object.values(posterior || {})
  const top = values.length ? Math.max(...values) : 0
  if (top >= 0.8) return 'This reading is a settled one — your recent sessions point the same way.'
  if (top >= 0.5) return 'This reading is a fairly confident one, based on your recent sessions.'
  return 'This reading is tentative — a session or two more will make it clearer.'
}

export default function TeachBack({ user }) {
  const [topics, setTopics] = useState(null)
  const [session, setSession] = useState(null)
  const [progress, setProgress] = useState(null) // timeline / concept_no / question_no
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [misconceptions, setMisconceptions] = useState([]) // {name, resolved}
  const [awaitingReport, setAwaitingReport] = useState(false)
  const [wrapPhase, setWrapPhase] = useState('summary') // summary -> feedback
  const [summary, setSummary] = useState('')
  const [report, setReport] = useState({ attention: 7, confidence: 5, difficulty: 5 })
  const [pace, setPace] = useState('About right')
  const [chips, setChips] = useState([])
  const [feedbackText, setFeedbackText] = useState('')
  const [result, setResult] = useState(null)
  const [quizOutcome, setQuizOutcome] = useState(null)
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
      setWrapPhase('summary')
      setSummary('')
      setPace('About right')
      setChips([])
      setFeedbackText('')
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
      const r = await api.finish(session.session_id, {
        ...report,
        summary: summary.trim(),
        pace,
        feedback_choices: chips,
        feedback_text: feedbackText.trim(),
      })
      setResult(r)
      setQuizOutcome(null)
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
    setQuizOutcome(null)
    setAwaitingReport(false)
  }

  /* ---------- topic selection ---------- */
  if (!session) {
    const bySubject = new Map()
    for (const t of topics || []) {
      const key = t.subject_name || 'Other topics'
      if (!bySubject.has(key)) bySubject.set(key, [])
      bySubject.get(key).push(t)
    }
    return (
      <div className="space-y-5">
        <div className="banner">TeachBack — What did you take away from the lecture?</div>
        {error && <div className="card p-4 text-sm text-red-700 bg-red-50 border-red-200">{error}</div>}
        <p className="text-sm text-charcoal-light">
          Pick the lecture/topic you want to explain. You don&apos;t need textbook definitions —
          <strong> your own words are exactly what we want</strong>. It&apos;s a short conversation,
          not an exam: simple questions, one at a time, and most answers only need one or two sentences.
        </p>
        {!topics && <div className="text-charcoal-light text-sm">Loading topics…</div>}
        {[...bySubject.entries()].map(([subjectName, list]) => (
          <div key={subjectName}>
            <div className="text-xs font-bold uppercase tracking-wide text-charcoal-light mb-2">{subjectName}</div>
            <div className="grid md:grid-cols-3 gap-4">
              {list.map((t) => (
                <button
                  key={t.id}
                  onClick={() => start(t.id)}
                  disabled={busy}
                  className="card p-5 text-left hover:border-brand hover:shadow-md transition-all group"
                >
                  <div className="font-bold text-charcoal group-hover:text-brand">{t.name}</div>
                  <p className="text-sm text-charcoal-light mt-1.5 line-clamp-2">{t.description}</p>
                  <div className="text-xs text-charcoal-light mt-3">
                    {t.concept_count} concepts · short conversational questions
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    )
  }

  /* ---------- session result ---------- */
  if (result) {
    const recommendation = quizOutcome?.updated_recommendation || result.recommendation
    const demonstrated = result.concept_summary.filter((c) => c.status === 'covered')
    const needsWork = result.concept_summary.filter((c) => c.status !== 'covered')
    // three distinct relationship states — "not discussed" is an absence of
    // evidence, shown neutrally and never mixed in with what needs clarifying
    const relsDemonstrated = (result.relationship_summary || []).filter((r) => r.status === 'demonstrated')
    const relsUnclear = (result.relationship_summary || []).filter((r) => r.status === 'needs_clarification')
    const relsNotDiscussed = (result.relationship_summary || []).filter((r) => r.status === 'not_discussed')
    return (
      <div className="space-y-5">
        <div className="banner">Session Complete — {session.topic.name}</div>

        <div className="card">
          <div className="card-header">Current learning condition</div>
          <div className="p-5">
            <div className="flex flex-col sm:flex-row sm:items-center gap-4">
              <StateBadge label={result.state.label} size="lg" audience="student" />
              <p className="text-sm text-charcoal-light">{studentStateDescription(result.state.label)}</p>
            </div>
            {result.state.note && (
              <p className="mt-3 text-sm text-charcoal bg-sky-50 border border-sky-200 rounded-md p-3">
                {result.state.note}
              </p>
            )}
            {result.previous_state_label && result.previous_state_label !== result.state.label && (
              <div className="mt-4 flex items-center gap-3 text-sm">
                <span className="text-xs font-semibold text-charcoal-light uppercase tracking-wide">Learning journey</span>
                <StateBadge label={result.previous_state_label} audience="student" />
                <span className="text-zinc-400">→</span>
                <StateBadge label={result.state.label} audience="student" />
              </div>
            )}
            {result.observation?.evidence?.length > 0 && (
              <div className="mt-4 pt-4 border-t border-zinc-100">
                <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-2">Why</div>
                <ul className="space-y-1">
                  {result.observation.evidence.map((e, i) => (
                    <li key={i} className="text-sm text-charcoal flex gap-2"><span className="text-brand">•</span>{e}</li>
                  ))}
                </ul>
              </div>
            )}
            {/* The five-state distribution is classifier output. A student is
                told in words how settled the reading is; the breakdown stays
                available but folded away, and is never presented as a
                probability that they understand the topic. */}
            <div className="mt-4 pt-4 border-t border-zinc-100">
              <p className="text-sm text-charcoal-light">{settledness(result.state.posterior)}</p>
              <details className="mt-2">
                <summary className="text-xs text-charcoal-light cursor-pointer hover:text-brand">
                  How this reading was made
                </summary>
                <p className="text-xs text-charcoal-light mt-2 mb-2">
                  {result.state.posterior_meaning ||
                    'How well each learning condition explains your run of sessions — not a probability that you understand the topic.'}
                </p>
                {Object.entries(result.state.posterior).map(([name, p]) => (
                  <Meter key={name} label={studentStateLabel(name)} value={p} color={name === result.state.label ? '#A5231B' : '#A1A1AA'} />
                ))}
              </details>
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
              {result.summary_insights?.new_concepts_demonstrated?.length > 0 && (
                <li className="pt-2 mt-1 border-t border-zinc-100 text-xs text-charcoal-light">
                  Your takeaway summary added evidence for:{' '}
                  <span className="text-charcoal font-medium">
                    {result.summary_insights.new_concepts_demonstrated.join(', ')}
                  </span>
                </li>
              )}
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

        {relsNotDiscussed.length > 0 && (
          <div className="card">
            <div className="card-header">Connections we didn&apos;t get to</div>
            <div className="p-4">
              <ul className="flex flex-wrap gap-x-6 gap-y-2">
                {relsNotDiscussed.map((r) => (
                  <li key={`${r.source}-${r.target}`} className="flex items-center gap-2 text-sm">
                    <span className="text-zinc-400">○</span>
                    <span className="text-charcoal">{r.source} <span className="text-zinc-400">→</span> {r.target}</span>
                  </li>
                ))}
              </ul>
              <p className="text-xs text-charcoal-light mt-3">
                Not discussed — these simply didn&apos;t come up in this conversation. They are not counted
                against you, and nothing here suggests you misunderstood them.
              </p>
            </div>
          </div>
        )}

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

        <KnowledgeCheck
          quizInfo={result.quiz}
          topicId={session.topic.id}
          studentId={user.id}
          sessionId={result.session_id}
          onResult={setQuizOutcome}
        />

        <div className="card">
          <div className="card-header">Recommended next activity</div>
          <div className="p-5">
            <div className="text-xs font-bold uppercase tracking-wide text-brand mb-1">
              {recommendation.activity.kind.replace('_', ' ')}
            </div>
            <div className="font-semibold text-charcoal">{recommendation.activity.title}</div>
            <p className="text-sm text-charcoal-light mt-1.5">{recommendation.activity.description}</p>
            {quizOutcome?.updated_recommendation && (
              <p className="text-xs text-charcoal-light mt-1.5 italic">
                Updated using your explanation and the knowledge check together.
              </p>
            )}
            {recommendation.why && (
              <div className="mt-3 pt-3 border-t border-zinc-100">
                <div className="text-xs font-semibold text-charcoal-light uppercase tracking-wide mb-1">Why this was recommended</div>
                <p className="text-sm text-charcoal">{recommendation.why}</p>
              </div>
            )}
            {recommendation.notes?.map((n, i) => (
              <div key={i} className="mt-3 text-xs bg-amber-50 border border-amber-200 text-amber-800 rounded-md p-2">{n}</div>
            ))}
            <Link
              to="/activity"
              state={{
                activity: { ...recommendation.activity, topic_id: session.topic.id, topic_name: session.topic.name },
                why: recommendation.why,
              }}
              className="btn-primary inline-block mt-4"
            >
              Start Activity →
            </Link>
          </div>
        </div>

        <div className="flex gap-3">
          <button onClick={reset} className="btn-secondary">Teach another topic</button>
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
            wrapPhase === 'summary' ? (
              <div className="border-t border-zinc-200 p-4 space-y-3">
                <div className="text-sm font-semibold text-charcoal">Your takeaway</div>
                <p className="text-xs text-charcoal-light">
                  In your own words, what did you learn from this lecture? A few sentences about what you
                  understood, what stood out, or how the ideas connect. This is your personal summary — not
                  an exam answer — and a short one is completely fine.
                </p>
                <textarea
                  className="input min-h-[110px] resize-y"
                  placeholder="What I took away from this lecture is…"
                  value={summary}
                  onChange={(e) => setSummary(e.target.value)}
                />
                <div className="flex items-center justify-between">
                  <button onClick={() => setWrapPhase('feedback')} className="text-xs text-charcoal-light hover:text-brand font-semibold uppercase tracking-wide">
                    Skip
                  </button>
                  <button onClick={() => setWrapPhase('feedback')} className="btn-primary">
                    Submit summary →
                  </button>
                </div>
              </div>
            ) : (
              <div className="border-t border-zinc-200 p-4 space-y-4">
                <div className="text-sm font-semibold text-charcoal">Last step — quick lecture feedback (under a minute)</div>
                <div>
                  <div className="text-xs text-charcoal-light mb-1.5">How did the lecture&apos;s pace feel?</div>
                  <div className="flex flex-wrap gap-1.5">
                    {PACE_OPTIONS.map((p) => (
                      <button
                        key={p}
                        onClick={() => setPace(p)}
                        className={`text-xs px-2.5 py-1.5 rounded border font-semibold transition-colors ${
                          pace === p ? 'bg-brand text-white border-brand' : 'bg-white text-charcoal-light border-zinc-300 hover:border-brand hover:text-brand'
                        }`}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
                {[
                  ['attention', 'How focused were you this session?'],
                  ['confidence', 'How confident do you feel about this material now?'],
                  ['difficulty', 'How difficult did this lecture feel?'],
                ].map(([key, label]) => (
                  <div key={key}>
                    <div className="flex justify-between text-xs text-charcoal-light mb-1">
                      <span>{label}</span>
                      <span className="font-bold text-charcoal tabular-nums">{report[key]}/10</span>
                    </div>
                    <input
                      type="range" min="0" max="10" step="1" value={report[key]}
                      onChange={(e) => setReport((r) => ({ ...r, [key]: Number(e.target.value) }))}
                      aria-label={label}
                      className="range"
                      style={{ '--range-frac': report[key] / 10 }}
                    />
                  </div>
                ))}
                <div>
                  <div className="text-xs text-charcoal-light mb-1.5">What could make this lecture better for you? (optional)</div>
                  <div className="flex flex-wrap gap-1.5">
                    {FEEDBACK_CHIPS.map((c) => (
                      <button
                        key={c}
                        onClick={() => setChips((cs) => (cs.includes(c) ? cs.filter((x) => x !== c) : [...cs, c]))}
                        className={`text-xs px-2.5 py-1.5 rounded border transition-colors ${
                          chips.includes(c) ? 'bg-brand text-white border-brand' : 'bg-white text-charcoal-light border-zinc-300 hover:border-brand hover:text-brand'
                        }`}
                      >
                        {c}
                      </button>
                    ))}
                  </div>
                  <input
                    className="input mt-2"
                    placeholder="Anything else you want your teacher to know? (optional)"
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                  />
                </div>
                <div className="flex gap-3">
                  <button onClick={() => setWrapPhase('summary')} className="btn-secondary">← Back</button>
                  <button onClick={finish} disabled={busy} className="btn-primary flex-1">
                    {busy ? 'Estimating your learning state…' : 'See my session summary'}
                  </button>
                </div>
              </div>
            )
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
            <strong className="text-charcoal">How this works:</strong> this is a short check of what you took
            away from the lecture — not an exam. Your answers are compared with what was actually taught, and
            once you&apos;ve shown you understood an idea, we move on. Your own words are enough; there are no
            wrong-answer penalties.
          </div>
        </div>
      </div>
    </div>
  )
}
