"use client";
import Link from "next/link";
import { useState, useEffect } from "react";
import LiveKitInterview from "../components/LiveKitInterview";
import ResumeUploader from "../components/ResumeUploader";
import { API_BASE } from "../utils/api";

interface Stats { candidates: number; interviews: number }
interface InterviewState {
  active: boolean;
  roomName?: string;
  token?: string;
  candidateName?: string;
  identity?: string;
  url?: string;
}

export default function Home() {
  const [interviewState, setInterviewState] = useState<InterviewState>({ active: false });
  const [stats, setStats] = useState<Stats>({ candidates: 0, interviews: 0 });

  useEffect(() => {
    // Load real stats from API (replaces hardcoded 12 / 3 / 1)
    Promise.all([
      fetch(`${API_BASE}/candidates`).then(r => r.ok ? r.json() : []),
      fetch(`${API_BASE}/interviews`).then(r => r.ok ? r.json() : []),
    ]).then(([candidates, interviews]) => {
      setStats({
        candidates: Array.isArray(candidates) ? candidates.length : 0,
        interviews: Array.isArray(interviews) ? interviews.length : 0,
      });
    }).catch(() => {});
  }, []);

  if (interviewState.active && interviewState.roomName && interviewState.token) {
    return (
      <LiveKitInterview
        roomName={interviewState.roomName}
        token={interviewState.token}
        serverUrl={interviewState.url}
        candidateName={interviewState.candidateName ?? "Candidate"}
        onLeave={() => setInterviewState({ active: false })}
      />
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Interview AI Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-sm font-medium text-slate-500">Total Candidates</h3>
          <p className="text-3xl font-bold mt-2">{stats.candidates}</p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-sm font-medium text-slate-500">Total Interviews</h3>
          <p className="text-3xl font-bold mt-2 text-blue-600">{stats.interviews}</p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-sm font-medium text-slate-500">System Status</h3>
          <p className="text-3xl font-bold mt-2 text-green-600">Live</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
        <div>
          <h2 className="text-lg font-semibold mb-4">Start New Interview</h2>
          <ResumeUploader
            onInterviewStart={(roomName, token, candidateName, identity, url) =>
              setInterviewState({ active: true, roomName, token, candidateName, identity, url })
            }
          />
        </div>
        <div>
          <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
          <div className="flex flex-col gap-3">
            <Link href="/candidates" className="block w-full p-4 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors">
              <span className="font-medium text-slate-700 block">View All Candidates</span>
              <span className="text-sm text-slate-500">Browse and manage applications</span>
            </Link>
            <Link href="/interviews" className="block w-full p-4 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors">
              <span className="font-medium text-slate-700 block">Interview History</span>
              <span className="text-sm text-slate-500">View past sessions and evaluations</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
