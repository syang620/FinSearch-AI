#!/usr/bin/env python3
"""
FinSearch AI CLI Interface
Interactive command-line interface for financial Q&A with RAG support
"""

import sys
import os
sys.path.insert(0, '.')

from typing import Optional
import argparse
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Import services
from app.services.llm.ollama_service import get_ollama_service
from app.services.rag.retriever import rag_retriever

console = Console()

class FinSearchCLI:
    def __init__(self, use_context: bool = True, company_filter: Optional[str] = None):
        """Initialize CLI with Ollama service"""
        self.use_context = use_context
        self.company_filter = company_filter
        self.ollama_service = None

        try:
            console.print("[cyan]Initializing FinSearch AI CLI...[/cyan]")
            self.ollama_service = get_ollama_service()
            model_name = self.ollama_service.model_name
            console.print(f"[green]✓ Connected to Ollama (Model: {model_name})[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to Ollama: {e}[/red]")
            console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
            sys.exit(1)

    def get_response(self, query: str) -> str:
        """Get response from LLM with optional RAG context"""
        context = None
        contexts_list = []

        if self.use_context:
            try:
                # Retrieve relevant context
                results = rag_retriever.retrieve_context(
                    query=query,
                    n_results=3,
                    company_filter=self.company_filter
                )

                if results and 'contexts' in results:
                    contexts_list = results['contexts']
                    context_texts = [ctx['text'] for ctx in contexts_list[:3]]
                    context = "\n\n".join(context_texts)

            except Exception as e:
                console.print(f"[yellow]Warning: Could not retrieve context: {e}[/yellow]")

        # Get response from Ollama
        response = self.ollama_service.chat(
            query=query,
            context=context,
            max_new_tokens=512
        )

        return response, contexts_list

    def run_interactive(self):
        """Run interactive chat session"""
        console.print(Panel.fit(
            "[bold cyan]FinSearch AI - Financial Q&A System[/bold cyan]\n"
            f"Mode: {'RAG-Enhanced' if self.use_context else 'Direct LLM'}\n"
            f"Company Filter: {self.company_filter or 'All Companies'}\n"
            "Type 'exit' or 'quit' to end session\n"
            "Type '/help' for commands",
            title="Welcome"
        ))

        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold green]You[/bold green]")

                # Check for exit commands
                if query.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                # Check for special commands
                if query == '/help':
                    self.show_help()
                    continue
                elif query == '/toggle':
                    self.use_context = not self.use_context
                    console.print(f"[cyan]Context mode: {'ON' if self.use_context else 'OFF'}[/cyan]")
                    continue
                elif query.startswith('/company'):
                    parts = query.split()
                    if len(parts) > 1:
                        self.company_filter = parts[1].upper()
                        console.print(f"[cyan]Company filter set to: {self.company_filter}[/cyan]")
                    else:
                        self.company_filter = None
                        console.print("[cyan]Company filter cleared[/cyan]")
                    continue

                # Get and display response
                console.print("\n[bold blue]FinSearch AI[/bold blue] [dim](thinking...)[/dim]")
                response, contexts = self.get_response(query)

                # Clear thinking message and show response
                console.print("\033[1A\033[2K", end='')  # Clear previous line
                console.print("[bold blue]FinSearch AI:[/bold blue]")

                # Display response with markdown formatting
                console.print(Markdown(response))

                # Show sources if available
                if contexts and self.use_context:
                    console.print("\n[dim]Sources:[/dim]")
                    for i, ctx in enumerate(contexts[:3], 1):
                        metadata = ctx.get('metadata', {})
                        # Try source_uri first, then filename, then source
                        source = metadata.get('source_uri') or metadata.get('filename') or metadata.get('source', 'Unknown')
                        # Extract just the filename from path if it's a full path
                        if '/' in source:
                            source = source.split('/')[-1]
                        console.print(f"  [{i}] {source}", style="dim")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def show_help(self):
        """Display help information"""
        help_text = """
[bold]Available Commands:[/bold]
  /help          - Show this help message
  /toggle        - Toggle context/RAG mode on/off
  /company TICK  - Filter by company ticker (e.g., /company AAPL)
  /company       - Clear company filter
  exit/quit      - Exit the CLI

[bold]Example Questions:[/bold]
  • What was Apple's revenue last quarter?
  • Explain EBITDA margin
  • Compare Microsoft and Google's operating margins
  • What are the key risks mentioned in Tesla's 10-K?
        """
        console.print(Panel(help_text, title="Help"))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FinSearch AI CLI')
    parser.add_argument('--no-context', action='store_true',
                       help='Disable RAG context retrieval')
    parser.add_argument('--company', type=str,
                       help='Filter by company ticker (e.g., AAPL)')
    parser.add_argument('query', nargs='*',
                       help='Direct query (non-interactive mode)')

    args = parser.parse_args()

    # Initialize CLI
    cli = FinSearchCLI(
        use_context=not args.no_context,
        company_filter=args.company.upper() if args.company else None
    )

    if args.query:
        # Non-interactive mode - single query
        query = ' '.join(args.query)
        response, contexts = cli.get_response(query)
        console.print(Markdown(response))
        if contexts and not args.no_context:
            console.print("\n[dim]Sources:[/dim]")
            for ctx in contexts[:3]:
                metadata = ctx.get('metadata', {})
                # Try source_uri first, then filename, then source
                source = metadata.get('source_uri') or metadata.get('filename') or metadata.get('source', 'Unknown')
                # Extract just the filename from path if it's a full path
                if '/' in source:
                    source = source.split('/')[-1]
                console.print(f"  • {source}", style="dim")
    else:
        # Interactive mode
        cli.run_interactive()

if __name__ == "__main__":
    main()