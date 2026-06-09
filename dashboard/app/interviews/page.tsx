"use client";
import { useEffect, useState } from 'react';
import { fetcher } from '@/utils/api';
import Link from 'next/link';
import { format } from 'date-fns';

export default function InterviewsPage() {
  const [interviews, setInterviews] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetcher<any[]>('/interviews').then(setInterviews).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Interviews</h1>

      <div className="grid gap-4">
        {interviews.map((i) => (
          <Link key={i.id} href={`/interviews/${i.id}`}>
            <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 hover:border-blue-400 transition-colors flex justify-between items-center group">
              <div>
                <h3 className="font-semibold text-slate-900">{i.room_name || "Untitled Interview"}</h3>
                <p className="text-sm text-slate-500">
                  {i.scheduled_time ? format(new Date(i.scheduled_time), 'PP p') : 'Unscheduled'}
                </p>
                <div className="mt-2 flex gap-2">
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700">
                    {i.status}
                  </span>
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600">
                    Room: {i.room_name}
                  </span>
                </div>
              </div>
              <div className="text-slate-400 group-hover:text-blue-600">
                →
              </div>
            </div>
          </Link>
        ))}
        {interviews.length === 0 && (
          <p className="text-slate-400">No interviews scheduled.</p>
        )}
      </div>
    </div>
  );
}
