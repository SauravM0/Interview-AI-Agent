'use client';

import {
    LiveKitRoom,
    VideoConference,
    useLocalParticipant,
    useRoomContext,
} from '@livekit/components-react';
import '@livekit/components-styles';
import { DataPacket_Kind, RoomEvent, Track } from 'livekit-client';
import { useEffect, useMemo, useState } from 'react';

interface LiveKitInterviewProps {
    roomName: string;
    candidateName: string;
    token?: string;
    serverUrl?: string;
    onLeave?: () => void;
}

interface TranscriptEntry {
    id: string;
    speaker: string;
    text: string;
    timestamp: number;
}

function getRemoteParticipants(room: any): any[] {
    const participants = room?.remoteParticipants ?? room?.participants;
    if (!participants) {
        return [];
    }
    if (typeof participants.values === 'function') {
        return Array.from(participants.values());
    }
    return Object.values(participants);
}

function StatusBadge({ state, detail }: { state: string; detail?: string }) {
    const label = detail && state === 'error' ? detail : state;
    const colorClass = useMemo(() => {
        switch (state) {
            case 'connected':
                return 'bg-emerald-100 text-emerald-700 border-emerald-200';
            case 'listening':
                return 'bg-sky-100 text-sky-700 border-sky-200';
            case 'thinking':
                return 'bg-amber-100 text-amber-700 border-amber-200';
            case 'speaking':
                return 'bg-violet-100 text-violet-700 border-violet-200';
            case 'error':
                return 'bg-rose-100 text-rose-700 border-rose-200';
            default:
                return 'bg-slate-100 text-slate-700 border-slate-200';
        }
    }, [state]);

    return (
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${colorClass}`}>
            {label}
        </span>
    );
}

function MicLevelMeter({ level }: { level: number }) {
    const percent = Math.min(100, Math.max(0, Math.round(level * 140)));
    return (
        <div className="flex items-center gap-3">
            <div className="text-xs text-slate-500 uppercase tracking-widest">Mic</div>
            <div className="flex-1 h-2 rounded-full bg-slate-200 overflow-hidden">
                <div
                    className="h-full bg-emerald-500 transition-all duration-100"
                    style={{ width: `${percent}%` }}
                />
            </div>
        </div>
    );
}

function InterviewRoomUI({ candidateName }: { candidateName: string }) {
    const room = useRoomContext();
    const { localParticipant } = useLocalParticipant();
    const [micLevel, setMicLevel] = useState(0);
    const [status, setStatus] = useState<{ state: string; detail?: string }>({ state: 'connected' });
    const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
    const [toast, setToast] = useState<string | null>(null);
    const [aiIdentity, setAiIdentity] = useState<string | null>(null);
    const [sendingEndTurn, setSendingEndTurn] = useState(false);

    useEffect(() => {
        const interval = window.setInterval(() => {
            setMicLevel(localParticipant?.audioLevel ?? 0);
        }, 120);
        return () => window.clearInterval(interval);
    }, [localParticipant]);

    useEffect(() => {
        if (!room) {
            return;
        }

        const handleData = (payload: Uint8Array, _participant: any, kind?: DataPacket_Kind, _topic?: string) => {
            if (kind === DataPacket_Kind.LOSSY) {
                return;
            }
            try {
                const decoded = JSON.parse(new TextDecoder().decode(payload));
                if (decoded?.type === 'transcript') {
                    setTranscripts((prev) => [
                        ...prev,
                        {
                            id: `${decoded.timestamp}-${prev.length}`,
                            speaker: decoded.speaker || 'Unknown',
                            text: decoded.text || '',
                            timestamp: decoded.timestamp || Date.now() / 1000,
                        },
                    ]);
                }
                if (decoded?.type === 'status') {
                    setStatus({ state: decoded.state || 'connected', detail: decoded.detail });
                    if (decoded.state === 'error' && decoded.detail) {
                        setToast(decoded.detail);
                    }
                }
            } catch (error) {
                console.error('Failed to parse data message', error);
            }
        };

        const handleParticipantConnected = (participant: any) => {
            if (!participant.isLocal && !aiIdentity) {
                setAiIdentity(participant.identity);
            }
        };

        const handleParticipantDisconnected = (participant: any) => {
            if (participant.identity === aiIdentity) {
                setStatus({ state: 'error', detail: 'AI disconnected' });
                setToast('AI disconnected from the room.');
            }
        };

        room.on(RoomEvent.DataReceived, handleData);
        room.on(RoomEvent.ParticipantConnected, handleParticipantConnected);
        room.on(RoomEvent.ParticipantDisconnected, handleParticipantDisconnected);

        const existingAi = getRemoteParticipants(room).find((participant: any) => !participant.isLocal);
        if (existingAi && !aiIdentity) {
            setAiIdentity(existingAi.identity);
        }

        return () => {
            room.off(RoomEvent.DataReceived, handleData);
            room.off(RoomEvent.ParticipantConnected, handleParticipantConnected);
            room.off(RoomEvent.ParticipantDisconnected, handleParticipantDisconnected);
        };
    }, [room, aiIdentity]);

    useEffect(() => {
        if (!room) {
            return;
        }

        const timeout = window.setTimeout(() => {
            const remoteParticipants = getRemoteParticipants(room);
            const hasAudio = remoteParticipants.some((participant: any) => {
                const micPub = participant.getTrackPublication(Track.Source.Microphone);
                return Boolean(micPub?.track);
            });
            if (!hasAudio) {
                setToast('AI audio track not detected. Please reconnect or refresh.');
                setStatus({ state: 'error', detail: 'AI audio not available' });
            }
        }, 8000);

        return () => window.clearTimeout(timeout);
    }, [room, transcripts.length]);

    useEffect(() => {
        if (!toast) {
            return;
        }
        const timeout = window.setTimeout(() => setToast(null), 4000);
        return () => window.clearTimeout(timeout);
    }, [toast]);

    const handleEndTurn = async () => {
        if (!room) {
            return;
        }
        setSendingEndTurn(true);
        try {
            const payload = new TextEncoder().encode(JSON.stringify({ type: 'end_turn', timestamp: Date.now() }));
            await room.localParticipant.publishData(payload, { reliable: true, topic: 'agent-control' });
            setToast('End turn sent. The AI will respond shortly.');
        } catch (error) {
            console.error('Failed to send end turn', error);
            setToast('Failed to send End Turn. Please try again.');
        } finally {
            setSendingEndTurn(false);
        }
    };

    const renderedTranscripts = transcripts.length ? transcripts : [
        {
            id: 'empty',
            speaker: candidateName,
            text: 'Say hello to start the interview.',
            timestamp: Date.now() / 1000,
        },
    ];

    return (
        <div className="flex flex-col lg:flex-row h-full w-full gap-4 p-4 bg-slate-50">
            <div className="flex-1 min-h-[60vh] rounded-2xl overflow-hidden border border-slate-200 bg-white shadow-sm">
                <VideoConference />
            </div>
            <aside className="w-full lg:w-[360px] flex flex-col gap-4">
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold text-slate-700">AI Status</div>
                        <StatusBadge state={status.state} detail={status.detail} />
                    </div>
                    <MicLevelMeter level={micLevel} />
                    <div className="text-xs text-slate-400">You are connected as {candidateName}.</div>
                    <button
                        onClick={handleEndTurn}
                        disabled={sendingEndTurn}
                        className={`w-full rounded-lg px-3 py-2 text-xs font-semibold transition-colors ${
                            sendingEndTurn
                                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                                : 'bg-slate-900 text-white hover:bg-slate-800'
                        }`}
                    >
                        {sendingEndTurn ? 'Sending End Turn...' : 'End Turn'}
                    </button>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm flex-1 overflow-hidden">
                    <div className="text-sm font-semibold text-slate-700 mb-3">Live Transcript</div>
                    <div className="space-y-3 overflow-y-auto max-h-[55vh] pr-2">
                        {renderedTranscripts.map((entry) => (
                            <div key={entry.id} className="flex flex-col gap-1">
                                <div className="flex items-center justify-between text-xs text-slate-400">
                                    <span className="font-semibold text-slate-500">{entry.speaker}</span>
                                    <span>
                                        {new Date(entry.timestamp * 1000).toLocaleTimeString([], {
                                            hour: '2-digit',
                                            minute: '2-digit',
                                            second: '2-digit',
                                        })}
                                    </span>
                                </div>
                                <div className="text-sm text-slate-700">{entry.text}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </aside>
            {toast && (
                <div className="fixed bottom-6 right-6 z-50 rounded-xl bg-slate-900 text-white px-4 py-3 text-sm shadow-lg">
                    {toast}
                </div>
            )}
        </div>
    );
}

export default function LiveKitInterview({ roomName, candidateName, token: providedToken, serverUrl, onLeave }: LiveKitInterviewProps) {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const [token, setToken] = useState<string>(providedToken || "");
    const [url, setUrl] = useState<string>(serverUrl || "");
    const [error, setError] = useState<string | null>(null);
    const [lastErrorDetail, setLastErrorDetail] = useState<string | null>(null);

    useEffect(() => {
        if (providedToken) {
            setToken(providedToken);
            if (serverUrl) {
                setUrl(serverUrl);
                setError(null);
            } else {
                setError("LiveKit server URL missing from backend response.");
            }
            return;
        }

        (async () => {
            try {
                const resp = await fetch(
                    `${API_BASE}/api/interviews/generate-token?room_name=${roomName}&identity=${candidateName}&name=${candidateName}`
                );
                if (!resp.ok) {
                    throw new Error(`Token request failed: ${resp.status} ${resp.statusText}`);
                }
                const data = await resp.json();
                if (!data?.token) {
                    setError("LiveKit token missing from backend.");
                    return;
                }
                setToken(data.token);
                if (!data?.url) {
                    setError("LiveKit server URL missing from backend.");
                    return;
                }
                setUrl(data.url);
                setError(null);
            } catch (e) {
                console.error(e);
                setError("Unable to contact LiveKit token service.");
                setLastErrorDetail(String(e));
            }
        })();
    }, [roomName, candidateName, providedToken, serverUrl]);

    const handleRoomError = (err: any) => {
        const msg = err?.message || err?.toString() || "Unable to connect to LiveKit.";
        const normalized = msg.toLowerCase();
        if (normalized.includes("invalid token") || normalized.includes("401")) {
            setError(`LiveKit rejected the token for room ${roomName}. Check LIVEKIT_API_SECRET matches livekit.yaml keys.`);
        } else {
            setError(msg);
        }
        setLastErrorDetail(msg);
    };

    if (error || !token || !url) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="text-center space-y-2">
                    <p className="text-lg font-semibold">{error || "Connecting to secure interview room..."}</p>
                    <p className="text-sm text-slate-300">Room: {roomName}</p>
                    {lastErrorDetail && (
                        <p className="text-xs text-slate-400 break-all">Detail: {lastErrorDetail}</p>
                    )}
                </div>
            </div>
        );
    }

    return (
        <LiveKitRoom
            video={true}
            audio={true}
            token={token}
            serverUrl={url}
            data-lk-theme="default"
            style={{ height: '100vh' }}
            onDisconnected={onLeave}
            onError={handleRoomError}
        >
            <InterviewRoomUI candidateName={candidateName} />
        </LiveKitRoom>
    );
}
