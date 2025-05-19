"""
Simple Main Entry Point for the Interview Scheduler.
Handles email processing, interview scheduling, and candidate insights.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from simple_interview_manager import SimpleInterviewManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interview_scheduler.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("InterviewScheduler")

def run_scheduler():
    """Run the interview scheduler in daemon mode."""
    try:
        logger.info("Starting Interview Scheduler in daemon mode...")
        
        # Create and run the interview manager
        manager = SimpleInterviewManager()
        manager.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

def query_candidate_insights(query):
    """Query candidate insights using natural language."""
    try:
        logger.info(f"Querying candidate insights: {query}")
        
        # Create the interview manager
        manager = SimpleInterviewManager()
        
        # Get insights
        result = manager.get_candidate_insights(query)
        
        if result['success']:
            print("\n=== Candidate Insights ===\n")
            print(result['response'])
            
            if result['sources']:
                print("\n=== Sources ===\n")
                for i, source in enumerate(result['sources'], 1):
                    print(f"Source {i} (Relevance: {source['score']:.2f})")
                    print(f"Candidate: {source['candidate']}")
                    print(f"Email: {source['email']}")
                    print(f"From: {source['source_file']}")
                    print(f"Content: {source['content']}")
                    print("-" * 80 + "\n")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error querying candidate insights: {e}")
        print(f"An error occurred: {str(e)}")

def main():
    """Main function to handle command-line arguments."""
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ['EMAIL_ADDRESS', 'EMAIL_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please create a .env file with the required variables.")
        return
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Interview Scheduler with RAG capabilities')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Daemon mode command
    daemon_parser = subparsers.add_parser('daemon', help='Run the interview scheduler in daemon mode')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query candidate insights')
    query_parser.add_argument('query', nargs='?', help='Natural language query about candidates')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'daemon':
        run_scheduler()
    elif args.command == 'query':
        if not args.query:
            # Interactive query mode
            try:
                print("Enter your query about candidates (or 'exit' to quit):")
                while True:
                    query = input("\nQuery: ").strip()
                    if query.lower() in ['exit', 'quit', 'q']:
                        break
                    if query:
                        query_candidate_insights(query)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
        else:
            query_candidate_insights(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
