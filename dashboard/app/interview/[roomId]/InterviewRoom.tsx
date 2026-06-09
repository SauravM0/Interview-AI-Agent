"use client";

import {
    LiveKitRoom,
    VideoConference,
    PreJoin,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { useEffect, useState } from "react";

interface InterviewRoomProps {
    roomId: string;
    url: string;
}

export function InterviewRoom({ roomId, url }: InterviewRoomProps) {
    const [token, setToken] = useState("");
    const [shouldConnect, setShouldConnect] = useState(false);
    const [interviewActive, setInterviewActive] = useState(false);

    useEffect(() => {
        (async () => {
            try {
                const resp = await fetch(
                    `/api/livekit/token?roomName=${roomId}&participantName=Candidate`
                );
                const data = await resp.json();
                setToken(data.token);
            } catch (e) {
                console.error(e);
            }
        })();
    }, [roomId]);

    if (!token) {
        return (
            <div className="flex items-center justify-center h-screen bg-black text-white">
                Loading...
            </div>
        );
    }

    // Pre-join screen
    if (!shouldConnect) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900">
                <div className="bg-white p-8 rounded-lg shadow-xl max-w-md w-full">
                    <h1 className="text-2xl font-bold mb-4 text-center text-gray-800">Join Interview</h1>
                    <PreJoin
                        onError={(err) => console.error("error while setting up prejoin", err)}
                        defaults={{
                            audioDeviceId: "",
                            videoDeviceId: "",
                        }}
                        onSubmit={(values) => {
                            setShouldConnect(true);
                        }}
                    />
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
            connect={shouldConnect}
            onConnected={() => setInterviewActive(true)}
            data-lk-theme="default"
            style={{ height: '100vh' }}
        >
            <VideoConference />
        </LiveKitRoom>
    );
}
