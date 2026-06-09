"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Users, FileText, LayoutDashboard, Settings } from 'lucide-react';
import { clsx } from 'clsx';

export default function Navbar() {
  const pathname = usePathname();

  const navs = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Candidates', href: '/candidates', icon: Users },
    { name: 'Interviews', href: '/interviews', icon: FileText },
  ];

  return (
    <nav className="w-64 min-h-screen bg-slate-900 text-white flex flex-col p-4">
      <div className="mb-8 p-2">
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Recruitment AI
        </h1>
        <p className="text-xs text-slate-400">Admin Portal</p>
      </div>

      <div className="space-y-2 flex-1">
        {navs.map((n) => {
          const Icon = n.icon;
          const isActive = pathname === n.href;
          return (
            <Link
              key={n.name}
              href={n.href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                isActive ? "bg-blue-600 text-white" : "text-slate-300 hover:bg-slate-800"
              )}
            >
              <Icon size={18} />
              <span className="text-sm font-medium">{n.name}</span>
            </Link>
          );
        })}
      </div>
      
      <div className="pt-4 border-t border-slate-700">
         <div className="flex items-center gap-2 p-2 text-slate-400 text-xs">
            <Settings size={14}/>
            <span>v1.0.0</span>
         </div>
      </div>
    </nav>
  );
}
