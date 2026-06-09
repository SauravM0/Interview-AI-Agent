"use client";

import {
    LiveKitRoom,
    VideoConference,
    PreJoin,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { useState } from "react";

interface ActiveInterviewRoomProps {
    token: string;
    serverUrl: string;
    candidateName: string;
}

export function ActiveInterviewRoom({ token, serverUrl, candidateName }: ActiveInterviewRoomProps) {
    const [shouldConnect, setShouldConnect] = useState(false);
    const [interviewActive, setInterviewActive] = useState(false);

    // Pre-join screen
    if (!shouldConnect) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-50">
                <div className="bg-white p-8 rounded-lg shadow-xl max-w-md w-full">
                    <h1 className="text-2xl font-bold mb-4 text-center text-gray-800">
                        Join Interview for {candidateName}
                    </h1>
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
            serverUrl={serverUrl}
            connect={shouldConnect}
            onConnected={() => setInterviewActive(true)}
            data-lk-theme="default"
            style={{ height: '100vh' }}
        >
            <VideoConference />
        </LiveKitRoom>
    );
}
