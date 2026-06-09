import { ActiveInterviewRoom } from '@/components/ActiveInterviewRoom';
import { InterviewReport } from '@/components/InterviewReport';

interface PageProps {
    params: Promise<{ id: string }>;
}

async function getInterview(id: string) {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const res = await fetch(`${apiUrl}/api/interviews/${id}`, { cache: 'no-store' });
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

    const roomName = interview.room_name || `interview-${id}`;
    const participantName = interview.candidate_name || "Candidate";
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const tokenRes = await fetch(
        `${apiUrl}/api/interviews/generate-token?room_name=${encodeURIComponent(roomName)}&identity=${encodeURIComponent(participantName)}&name=${encodeURIComponent(participantName)}`,
        { cache: 'no-store' }
    );

    if (!tokenRes.ok) {
        return <div>Error: unable to create LiveKit interview token</div>;
    }

    const tokenData = await tokenRes.json();

    return (
        <ActiveInterviewRoom
            token={tokenData.token}
            serverUrl={tokenData.url}
            candidateName={participantName}
        />
    );
}
