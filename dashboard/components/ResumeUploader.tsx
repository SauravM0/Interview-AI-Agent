"use client";

import { useState } from 'react';

const API_Base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UploadResponse {
    success: boolean;
    room_name: string;
    token: string;
    url: string;
    identity: string;
    candidate_name: string;
}

interface ResumeUploaderProps {
    onInterviewStart: (roomName: string, token: string, candidateName: string, identity: string, url: string) => void;
}

export default function ResumeUploader({ onInterviewStart }: ResumeUploaderProps) {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [successData, setSuccess] = useState<UploadResponse | null>(null);
    const [settingsError, setSettingsError] = useState<string | null>(null);
    const [savingSettings, setSavingSettings] = useState(false);
    const [settings, setSettings] = useState({
        targetRole: "",
        experienceLevel: "",
        interviewType: "",
        focusAreas: [] as string[],
        durationMinutes: "",
        preferredLanguage: "English",
    });

    const focusOptions = [
        "Projects",
        "DSA",
        "System design",
        "JS/React",
        "Python",
        "Backend",
        "Frontend",
        "Leadership",
    ];

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
            setSuccess(null);
            setSettingsError(null);
            setSettings({
                targetRole: "",
                experienceLevel: "",
                interviewType: "",
                focusAreas: [],
                durationMinutes: "",
                preferredLanguage: "English",
            });
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${API_Base}/api/resume/upload`, {
                method: 'POST',
                body: formData, // Browser sets Content-Type to multipart/form-data automatically
            });

            if (!res.ok) {
                let message = `${res.status} ${res.statusText}`.trim();
                try {
                    const err = await res.json();
                    const detail = err?.detail ?? err?.message;
                    if (typeof detail === "string") {
                        message = detail;
                    } else if (detail && typeof detail === "object") {
                        message = detail.message || JSON.stringify(detail);
                    } else if (err) {
                        message = err.message || JSON.stringify(err);
                    }
                    if (res.status === 400 && message) {
                        message = `Upload failed: ${message}`;
                    }
                } catch {
                    // Fall back to status + statusText if JSON parsing fails.
                }
                throw new Error(message || "Upload failed");
            }

            const data: UploadResponse = await res.json();
            setSuccess(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const toggleFocusArea = (area: string) => {
        setSettings((prev) => {
            const hasArea = prev.focusAreas.includes(area);
            return {
                ...prev,
                focusAreas: hasArea ? prev.focusAreas.filter((item) => item !== area) : [...prev.focusAreas, area],
            };
        });
    };

    const canStartInterview = Boolean(
        settings.targetRole &&
        settings.experienceLevel &&
        settings.interviewType &&
        settings.durationMinutes &&
        settings.preferredLanguage
    );

    const handleStartInterview = async () => {
        if (!successData || !canStartInterview) {
            setSettingsError("Please complete all required fields before starting the interview.");
            return;
        }

        setSavingSettings(true);
        setSettingsError(null);

        try {
            const payload = {
                room_name: successData.room_name,
                target_role: settings.targetRole,
                experience_level: settings.experienceLevel,
                interview_type: settings.interviewType,
                focus_areas: settings.focusAreas,
                duration_minutes: Number(settings.durationMinutes),
                preferred_language: settings.preferredLanguage,
            };

            const res = await fetch(`${API_Base}/api/interviews/settings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                let message = res.statusText;
                try {
                    const err = await res.json();
                    message = err.detail || err.message || JSON.stringify(err);
                } catch {
                    // Ignore JSON parse errors and fallback to statusText.
                }
                throw new Error(`${res.status} ${message}`.trim());
            }

            onInterviewStart(successData.room_name, successData.token, successData.candidate_name, successData.identity, successData.url);
        } catch (err: any) {
            setSettingsError(err.message || "Failed to save interview settings.");
        } finally {
            setSavingSettings(false);
        }
    };

    if (successData) {
        return (
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm animate-fade-in space-y-6">
                <div>
                    <h3 className="text-xl font-bold text-slate-800 mb-2">Resume Analyzed!</h3>
                    <p className="text-slate-600">
                        We have processed your resume for <strong>{successData.candidate_name}</strong>.
                        <br />
                        Please complete the interview setup before joining the room.
                    </p>
                </div>

                <div className="grid gap-4">
                    <label className="text-sm font-medium text-slate-700">
                        Target role
                        <input
                            type="text"
                            value={settings.targetRole}
                            onChange={(e) => setSettings((prev) => ({ ...prev, targetRole: e.target.value }))}
                            className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                            placeholder="e.g., Frontend Engineer"
                            required
                        />
                    </label>

                    <label className="text-sm font-medium text-slate-700">
                        Experience level
                        <select
                            value={settings.experienceLevel}
                            onChange={(e) => setSettings((prev) => ({ ...prev, experienceLevel: e.target.value }))}
                            className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                            required
                        >
                            <option value="">Select level</option>
                            <option value="0-1">0-1 years</option>
                            <option value="1-3">1-3 years</option>
                            <option value="3-5">3-5 years</option>
                            <option value="5+">5+ years</option>
                        </select>
                    </label>

                    <label className="text-sm font-medium text-slate-700">
                        Interview type
                        <select
                            value={settings.interviewType}
                            onChange={(e) => setSettings((prev) => ({ ...prev, interviewType: e.target.value }))}
                            className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                            required
                        >
                            <option value="">Select type</option>
                            <option value="HR">HR</option>
                            <option value="Technical">Technical</option>
                            <option value="Behavioral">Behavioral</option>
                            <option value="Mixed">Mixed</option>
                        </select>
                    </label>

                    <div className="text-sm font-medium text-slate-700">
                        Focus areas
                        <div className="mt-2 grid grid-cols-2 gap-2">
                            {focusOptions.map((area) => (
                                <label
                                    key={area}
                                    className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs cursor-pointer ${
                                        settings.focusAreas.includes(area)
                                            ? "border-indigo-300 bg-indigo-50 text-indigo-700"
                                            : "border-slate-200 text-slate-600"
                                    }`}
                                >
                                    <input
                                        type="checkbox"
                                        checked={settings.focusAreas.includes(area)}
                                        onChange={() => toggleFocusArea(area)}
                                    />
                                    {area}
                                </label>
                            ))}
                        </div>
                    </div>

                    <label className="text-sm font-medium text-slate-700">
                        Duration
                        <select
                            value={settings.durationMinutes}
                            onChange={(e) => setSettings((prev) => ({ ...prev, durationMinutes: e.target.value }))}
                            className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                            required
                        >
                            <option value="">Select duration</option>
                            <option value="10">10 minutes</option>
                            <option value="20">20 minutes</option>
                            <option value="30">30 minutes</option>
                        </select>
                    </label>

                    <label className="text-sm font-medium text-slate-700">
                        Preferred language
                        <input
                            type="text"
                            value={settings.preferredLanguage}
                            onChange={(e) => setSettings((prev) => ({ ...prev, preferredLanguage: e.target.value }))}
                            className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
                            required
                        />
                    </label>
                </div>

                {settingsError && (
                    <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                        {settingsError}
                    </div>
                )}

                <div className="flex flex-col sm:flex-row gap-3">
                    <button
                        onClick={handleStartInterview}
                        disabled={!canStartInterview || savingSettings}
                        className={`flex-1 px-6 py-3 rounded-lg font-semibold shadow-sm transition-colors ${
                            !canStartInterview || savingSettings
                                ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                                : "bg-indigo-600 text-white hover:bg-indigo-700"
                        }`}
                    >
                        {savingSettings ? "Saving..." : "Start Interview Now"}
                    </button>
                    <button
                        onClick={() => setSuccess(null)}
                        className="flex-1 px-6 py-3 bg-white text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50 font-medium"
                    >
                        Not Now
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Upload Resume</h3>

            <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${file ? 'border-indigo-400 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400'}`}>
                <input
                    type="file"
                    id="resume-upload"
                    className="hidden"
                    accept=".pdf,.txt,.md"
                    onChange={handleFileChange}
                />

                {file ? (
                    <div>
                        <p className="text-indigo-700 font-medium truncate max-w-xs mx-auto">{file.name}</p>
                        <button
                            onClick={() => setFile(null)}
                            className="text-xs text-red-500 hover:underline mt-2"
                        >
                            Remove
                        </button>
                    </div>
                ) : (
                    <label htmlFor="resume-upload" className="cursor-pointer block">
                        <div className="mx-auto w-12 h-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center mb-3">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                        </div>
                        <p className="text-slate-600 font-medium">Click to upload resume</p>
                        <p className="text-xs text-slate-400 mt-1">PDF, TXT (Max 5MB)</p>
                    </label>
                )}
            </div>

            {error && (
                <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                    {error}
                </div>
            )}

            <button
                onClick={handleUpload}
                disabled={!file || loading}
                className={`w-full mt-4 py-2 rounded-lg font-semibold transition-colors ${!file || loading
                        ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                        : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm'
                    }`}
            >
                {loading ? "Processing..." : "Upload & Continue"}
            </button>
        </div>
    );
}
