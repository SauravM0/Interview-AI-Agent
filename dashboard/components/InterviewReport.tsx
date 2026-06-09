"use client";

import { useState, useEffect } from "react";
import { FileText, MessageSquare, Download } from "lucide-react";
import { fetcher } from "@/utils/api";

interface InterviewReportProps {
    data: any;
}

export function InterviewReport({ data }: InterviewReportProps) {
    const [activeTab, setActiveTab] = useState<'report' | 'transcript'>('report');
    const [report, setReport] = useState<string>("");
    const [loadingReport, setLoadingReport] = useState(true);

    useEffect(() => {
        if (data?.id) {
            fetcher<{ content: string }>(`/interviews/${data.id}/report`)
                .then((reportData) => {
                    setReport(reportData.content);
                })
                .catch(() => {
                    setReport("No report generated yet. (Complete interview to generate)");
                })
                .finally(() => setLoadingReport(false));
        }
    }, [data?.id]);


    return (
        <div className="space-y-6 max-w-5xl mx-auto">
            <div className="flex justify-between items-start">
                <div>
                    <h1 className="text-2xl font-bold">{data.candidate_name || "Unknown Candidate"}</h1>
                    <p className="text-slate-500">Interview: {data.room_name}</p>
                </div>
                <button className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg text-sm">
                    <Download size={16} /> Download PDF
                </button>
            </div>

            <div className="border-b border-slate-200">
                <nav className="-mb-px flex gap-6" aria-label="Tabs">
                    <button
                        onClick={() => setActiveTab('report')}
                        className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${activeTab === 'report'
                            ? 'border-blue-500 text-blue-600'
                            : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                            }`}
                    >
                        <FileText size={16} /> Evaluation Report
                    </button>
                    <button
                        onClick={() => setActiveTab('transcript')}
                        className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${activeTab === 'transcript'
                            ? 'border-blue-500 text-blue-600'
                            : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                            }`}
                    >
                        <MessageSquare size={16} /> Transcript
                    </button>
                </nav>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-sm border border-slate-200 min-h-[500px]">
                {activeTab === 'report' ? (
                    <article className="prose prose-slate max-w-none">
                        {/* We would render markdown here properly, for now just pre-wrap */}
                        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                            {loadingReport ? "Loading Report..." : report}
                        </pre>
                    </article>
                ) : (
                    <div className="space-y-4">
                        <p className="text-slate-400 italic text-center">Transcript view not fully implemented.</p>
                        {/* Iterate transcripts if available */}
                    </div>
                )}
            </div>
        </div>
    );
}
