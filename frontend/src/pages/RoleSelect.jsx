import { useEffect, useState } from 'react'
import { api } from '../services/api.js'
import StateBadge from '../components/StateBadge.jsx'

export default function RoleSelect({ onSelect }) {
  const [students, setStudents] = useState(null)
  const [showStudents, setShowStudents] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.students().then(setStudents).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <div className="bg-brand-darker text-white text-xs px-6 py-1.5 tracking-wide">
        Welcome to the TeachBack Learning Module
      </div>
      <div className="flex-1 flex items-center justify-center bg-zinc-100 p-6">
        <div className="w-full max-w-3xl">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-brand text-white rounded-xl flex items-center justify-center font-black text-2xl mx-auto mb-4 shadow">
              TB
            </div>
            <h1 className="text-3xl font-bold text-charcoal">TeachBack</h1>
            <p className="text-charcoal-light mt-2 max-w-lg mx-auto">
              Instead of rating how well you understood a topic, teach it back in your own words.
              The system listens, finds what you demonstrated, and tracks how your learning state evolves.
            </p>
          </div>

          {error && (
            <div className="card p-4 mb-4 text-sm text-red-700 bg-red-50 border-red-200">
              Could not reach the backend ({error}). Start it with <code className="font-mono">uvicorn app.main:app</code> and reload.
            </div>
          )}

          {!showStudents ? (
            <div className="grid md:grid-cols-2 gap-5">
              <button
                onClick={() => setShowStudents(true)}
                className="card p-8 text-left hover:border-brand hover:shadow-md transition-all group"
              >
                <div className="text-3xl mb-3">🎓</div>
                <div className="font-bold text-lg text-charcoal group-hover:text-brand">Continue as Student</div>
                <p className="text-sm text-charcoal-light mt-1">
                  Pick a demo student, choose a topic and teach it back.
                </p>
              </button>
              <button
                onClick={() => onSelect({ role: 'teacher', name: 'Prof. R. Deshpande', program: 'Computer Engineering' })}
                className="card p-8 text-left hover:border-brand hover:shadow-md transition-all group"
              >
                <div className="text-3xl mb-3">🧑‍🏫</div>
                <div className="font-bold text-lg text-charcoal group-hover:text-brand">Continue as Teacher</div>
                <p className="text-sm text-charcoal-light mt-1">
                  See the class state distribution, misconceptions and manage topics.
                </p>
              </button>
            </div>
          ) : (
            <div className="card">
              <div className="card-header">
                <span>Select a demo student</span>
                <button onClick={() => setShowStudents(false)} className="text-white/80 hover:text-white normal-case font-normal">
                  ← back
                </button>
              </div>
              <div className="p-4 grid sm:grid-cols-2 gap-2 max-h-96 overflow-y-auto">
                {!students && <div className="text-sm text-charcoal-light p-4">Loading students…</div>}
                {students?.filter((s) => s.is_demo).map((s) => (
                  <button
                    key={s.id}
                    onClick={() => onSelect({ role: 'student', id: s.id, name: s.name, program: s.program })}
                    className="flex items-center justify-between gap-3 p-3 rounded-md border border-zinc-200 hover:border-brand hover:bg-brand-light/40 transition-colors text-left"
                  >
                    <div>
                      <div className="font-semibold text-sm text-charcoal">{s.name}</div>
                      <div className="text-xs text-charcoal-light">
                        {s.program}{s.roll_no ? ` · ${s.roll_no}` : ''}
                      </div>
                    </div>
                    <StateBadge label={s.current_state_label} />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
