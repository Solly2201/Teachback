import { NavLink } from 'react-router-dom'

const studentNav = [
  { to: '/', label: 'Dashboard', icon: '⌂' },
  { to: '/teachback', label: 'TeachBack', icon: '✎' },
  { to: '/progress', label: 'Progress', icon: '↗' },
]

const teacherNav = [
  { to: '/', label: 'Class Overview', icon: '⌂' },
  { to: '/lectures', label: 'Lecture TeachBacks', icon: '▤' },
  { to: '/topics', label: 'Topic Management', icon: '☰' },
]

export default function Layout({ user, onSignOut, children }) {
  const nav = user.role === 'student' ? studentNav : teacherNav
  return (
    <div className="min-h-screen flex flex-col">
      {/* thin institutional strip */}
      <div className="bg-brand-darker text-white text-xs px-6 py-1.5 flex justify-between items-center">
        <span className="tracking-wide">Welcome to the TeachBack Learning Module</span>
        <span className="text-white/80">Academic Year 2026–27</span>
      </div>

      {/* main header */}
      <header className="bg-white border-b border-zinc-200 px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-brand text-white rounded-md flex items-center justify-center font-black text-lg select-none">
            TB
          </div>
          <div>
            <div className="font-bold text-lg text-charcoal leading-tight">TeachBack</div>
            <div className="text-xs text-charcoal-light">Teach it back. We&apos;ll listen.</div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="font-semibold text-sm text-charcoal">{user.name}</div>
            <div className="text-xs text-charcoal-light capitalize">{user.role}{user.program ? ` · ${user.program}` : ''}</div>
          </div>
          <div className="w-9 h-9 rounded-full bg-brand-light text-brand font-bold flex items-center justify-center border border-brand/20">
            {user.name.split(' ').map((p) => p[0]).slice(0, 2).join('')}
          </div>
          <button onClick={onSignOut} className="text-xs text-charcoal-light hover:text-brand font-semibold uppercase tracking-wide">
            Switch user
          </button>
        </div>
      </header>

      <div className="flex flex-1">
        {/* sidebar */}
        <aside className="w-52 bg-white border-r border-zinc-200 py-4 hidden md:block">
          <nav className="space-y-1 px-3">
            {nav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-semibold transition-colors ${
                    isActive
                      ? 'bg-brand text-white shadow-sm'
                      : 'text-charcoal-light hover:bg-zinc-100 hover:text-charcoal'
                  }`
                }
              >
                <span className="text-base w-4 text-center">{item.icon}</span>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main className="flex-1 p-6 max-w-6xl mx-auto w-full">{children}</main>
      </div>

      <footer className="text-center text-xs text-charcoal-light py-4 border-t border-zinc-200 bg-white">
        TeachBack · NLP + Hidden Markov Model mini-project · Department of Computer Engineering
      </footer>
    </div>
  )
}
