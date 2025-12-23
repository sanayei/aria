#!/usr/bin/env python3
"""Gmail OAuth setup wizard for ARIA.

This script guides you through setting up Gmail integration with ARIA by:
1. Helping you create a Google Cloud project
2. Configuring OAuth consent screen
3. Downloading credentials
4. Running the OAuth flow
5. Testing the connection

Run this script before using Gmail-related features in ARIA.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import aria modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

from aria.config import get_settings
from aria.tools.email.auth import GmailAuth, GmailAuthError


console = Console()


def print_step(step_num: int, title: str):
    """Print a step header."""
    console.print(f"\n[bold cyan]Step {step_num}: {title}[/bold cyan]\n")


def print_instructions(md_text: str):
    """Print markdown instructions."""
    console.print(Markdown(md_text))


def main():
    """Run Gmail setup wizard."""
    console.print()
    console.print(
        Panel.fit(
            "[bold]Gmail OAuth Setup Wizard for ARIA[/bold]\n\n"
            "This wizard will help you set up Gmail integration.\n"
            "You'll need a Google account and about 10 minutes.",
            title="Welcome",
            border_style="cyan",
        )
    )

    # Check if already configured
    settings = get_settings()
    gmail_auth = GmailAuth(settings.gmail_credentials_dir)

    if gmail_auth.is_authenticated():
        console.print("\n[yellow]Gmail is already configured![/yellow]")
        if not Confirm.ask("Do you want to reconfigure?", default=False):
            console.print(
                "[green]Setup cancelled. Your existing configuration is unchanged.[/green]\n"
            )
            return

    # Step 1: Create Google Cloud Project
    print_step(1, "Create Google Cloud Project")
    print_instructions("""
You need to create a Google Cloud project and enable the Gmail API:

1. Go to: https://console.cloud.google.com/
2. Click **"Select a project"** → **"New Project"**
3. Name it something like "ARIA Gmail Integration"
4. Click **"Create"**
5. Wait for project creation (usually a few seconds)

Once created, select your new project from the dropdown.
    """)

    if not Confirm.ask("\n[bold]Have you created the project?[/bold]", default=False):
        console.print(
            "[yellow]Please create a project first, then run this script again.[/yellow]\n"
        )
        return

    # Step 2: Enable Gmail API
    print_step(2, "Enable Gmail API")
    print_instructions("""
Now enable the Gmail API for your project:

1. Go to: https://console.cloud.google.com/apis/library/gmail.googleapis.com
2. Make sure your project is selected in the dropdown at the top
3. Click **"Enable"**
4. Wait for the API to be enabled
    """)

    if not Confirm.ask("\n[bold]Have you enabled the Gmail API?[/bold]", default=False):
        console.print("[yellow]Please enable the Gmail API, then run this script again.[/yellow]\n")
        return

    # Step 3: Configure OAuth Consent Screen
    print_step(3, "Configure OAuth Consent Screen")
    print_instructions("""
Configure the OAuth consent screen:

1. Go to: https://console.cloud.google.com/apis/credentials/consent
2. Select **"External"** user type (unless you have a Google Workspace account)
3. Click **"Create"**
4. Fill in the required fields:
   - **App name**: "ARIA"
   - **User support email**: Your email
   - **Developer contact email**: Your email
5. Click **"Save and Continue"**
6. On the "Scopes" page, click **"Save and Continue"** (skip adding scopes)
7. On the "Test users" page:
   - Click **"Add Users"**
   - Add your Gmail address
   - Click **"Save and Continue"**
8. Review and click **"Back to Dashboard"**
    """)

    if not Confirm.ask(
        "\n[bold]Have you configured the OAuth consent screen?[/bold]", default=False
    ):
        console.print(
            "[yellow]Please configure OAuth consent, then run this script again.[/yellow]\n"
        )
        return

    # Step 4: Create OAuth Credentials
    print_step(4, "Create OAuth Credentials")
    print_instructions("""
Create OAuth 2.0 credentials:

1. Go to: https://console.cloud.google.com/apis/credentials
2. Click **"Create Credentials"** → **"OAuth client ID"**
3. Select **"Desktop app"** as the application type
4. Name it "ARIA Desktop"
5. Click **"Create"**
6. In the popup, click **"Download JSON"**
7. Save the file (it will be named something like `client_secret_xxxxx.json`)
    """)

    if not Confirm.ask("\n[bold]Have you downloaded the credentials file?[/bold]", default=False):
        console.print("[yellow]Please download credentials, then run this script again.[/yellow]\n")
        return

    # Step 5: Move Credentials File
    print_step(5, "Move Credentials File")

    # Get credentials file path from user
    while True:
        creds_file_path = Prompt.ask(
            "\n[bold]Enter the path to your downloaded credentials file[/bold]",
            default="~/Downloads/client_secret*.json",
        )

        # Expand path
        creds_file_path = Path(creds_file_path).expanduser()

        # Handle wildcards if provided
        if "*" in str(creds_file_path):
            import glob

            matches = glob.glob(str(creds_file_path))
            if matches:
                creds_file_path = Path(matches[0])
            else:
                console.print(f"[red]No files found matching: {creds_file_path}[/red]")
                continue

        if not creds_file_path.exists():
            console.print(f"[red]File not found: {creds_file_path}[/red]")
            if not Confirm.ask("Try again?", default=True):
                console.print("[yellow]Setup cancelled.[/yellow]\n")
                return
            continue

        break

    # Copy credentials to ARIA's credentials directory
    try:
        import shutil

        gmail_auth.credentials_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(creds_file_path, gmail_auth.credentials_path)
        console.print(f"\n[green]✓ Credentials copied to {gmail_auth.credentials_path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to copy credentials: {e}[/red]")
        return

    # Step 6: Run OAuth Flow
    print_step(6, "Complete OAuth Authorization")
    console.print("Now we'll open a browser window for you to authorize ARIA to access your Gmail.")
    console.print("[dim]You'll be asked to sign in and grant permissions.[/dim]\n")

    if not Confirm.ask("Ready to continue?", default=True):
        console.print("[yellow]Setup cancelled.[/yellow]\n")
        return

    # Run the OAuth flow
    try:
        console.print("\n[cyan]Opening browser for authorization...[/cyan]")
        asyncio.run(_run_auth_flow(gmail_auth))
    except GmailAuthError as e:
        console.print(f"\n[red]Authentication failed: {e}[/red]")
        console.print("\n[yellow]Please try running this script again.[/yellow]\n")
        return
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return

    # Step 7: Test Connection
    print_step(7, "Test Connection")
    console.print("Testing Gmail connection...\n")

    try:
        success = asyncio.run(gmail_auth.test_connection())
        if success:
            console.print()
            console.print(
                Panel.fit(
                    "[bold green]✓ Gmail setup complete![/bold green]\n\n"
                    "You can now use Gmail features in ARIA.\n"
                    "Try: [cyan]aria chat[/cyan] and ask ARIA to check your emails!",
                    title="Success",
                    border_style="green",
                )
            )
            console.print()
        else:
            console.print("[red]Connection test failed. Please check the errors above.[/red]\n")
    except Exception as e:
        console.print(f"[red]Connection test failed: {e}[/red]\n")


async def _run_auth_flow(gmail_auth: GmailAuth):
    """Run the OAuth flow.

    Args:
        gmail_auth: GmailAuth instance

    Raises:
        GmailAuthError: If authentication fails
    """
    await gmail_auth.authenticate(force_reauth=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Setup cancelled by user.[/yellow]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
