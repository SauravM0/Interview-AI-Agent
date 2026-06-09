import { AccessToken } from 'livekit-server-sdk';
import { ActiveInterviewRoom } from '@/components/ActiveInterviewRoom';
import { InterviewReport } from '@/components/InterviewReport';

interface PageProps {
    params: Promise<{ id: string }>;
}

async function getInterview(id: string) {
    const res = await fetch(`http://127.0.0.1:8000/api/interviews/${id}`, { cache: 'no-store' });
    if (!res.ok) {
        if (res.status === 404) return null;
        throw new Error(`Failed to fetch interview: ${res.statusText}`);
    }
    return res.json();
}

export default async function InterviewPage({ params }: PageProps) {
    const { id } = await params;
    const interview = await getInterview(id);

    if (!interview) {
        return <div>Interview not found</div>;
    }

    // If completed, show report
    // Note: Adjust 'COMPLETED' string based on actual enum if different (e.g. 'completed' lowercase)
    // Assuming 'COMPLETED' based on user request.
    if (interview.status === 'COMPLETED') {
        return <InterviewReport data={interview} />;
    }

    // Otherwise (PENDING, SCHEDULED, IN_PROGRESS), show the Active Room
    const apiKey = process.env.LIVEKIT_API_KEY;
    const apiSecret = process.env.LIVEKIT_API_SECRET;
    const livekitUrl = process.env.LIVEKIT_URL || "ws://localhost:7880";

    if (!apiKey || !apiSecret) {
        return <div>Error: LiveKit Server not configured</div>;
    }

    const roomName = interview.room_name || `interview-${id}`;
    const participantName = interview.candidate_name || "Candidate";

    const at = new AccessToken(apiKey, apiSecret, { identity: participantName });
    at.addGrant({
        roomJoin: true,
        room: roomName,
        canPublish: true,
        canSubscribe: true,
    });

    const token = await at.toJwt();

    return (
        <ActiveInterviewRoom
            token={token}
            serverUrl={livekitUrl}
            candidateName={participantName}
        />
    );
}
