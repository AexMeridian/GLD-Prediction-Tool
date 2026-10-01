# Run this once, in a normal (non-admin) PowerShell window, on the machine
# that hosts the Ubuntu WSL distro at /root/gold-edge.
#
# Registers three per-user Scheduled Tasks so the Gold Edge shadow processes
# (no real orders, never place trades) and the daily free/keyless history
# backfill keep running after a reboot without anyone needing to notice and
# restart them by hand:
#
#   GoldEdgeShadowWatchdog     - fires at logon, then every 15 minutes
#                                forever. Calls ops/start_shadow_if_needed.sh,
#                                which is safe to call repeatedly:
#                                shadow_supervisor.sh takes an exclusive
#                                flock as its first action, so a duplicate
#                                launch just exits immediately instead of
#                                running two shadow processes against the
#                                same sqlite file.
#   GoldEdgeShadowFullWatchdog - the same idea for the SECOND, parallel
#                                shadow track (learning/shadow_full.py) that
#                                reuses the real entry/exit state machine for
#                                multiple round trips per window, writing to
#                                its own data/shadow_full.sqlite. Calls
#                                ops/start_shadow_full_if_needed.sh /
#                                shadow_full_supervisor.sh, its own
#                                independent flock -- never competes with or
#                                blocks the original shadow watchdog above.
#   GoldEdgeDailyBackfill      - fires daily at 04:10, plus once at logon as
#                                a catch-up if the PC was off at 04:10. Calls
#                                ops/daily_backfill.sh (idempotent: INSERT OR
#                                IGNORE), since Kalshi's own public API only
#                                retains ~2 months of settled-market history.
#
# None of the tasks need admin rights to register (LogonType Interactive,
# RunLevel Limited, current user only) or to run. Safe to re-run this script
# any time (e.g. after this update) -- -Force overwrites each task's
# definition in place without affecting the other two.

$ErrorActionPreference = "Stop"

$wsl = (Get-Command wsl.exe).Source
$currentUser = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited

# --- Task 1: watchdog, at logon + every 15 min forever ---
$action1 = New-ScheduledTaskAction -Execute $wsl `
    -Argument "-d Ubuntu -u root -e bash /root/gold-edge/ops/start_shadow_if_needed.sh"
$trigger1a = New-ScheduledTaskTrigger -AtLogOn
$trigger1b = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings1 = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "GoldEdgeShadowWatchdog" -Action $action1 -Trigger @($trigger1a, $trigger1b) -Settings $settings1 -Principal $principal -Description "Keeps the Gold Edge shadow-mode (no real orders) process running in WSL; idempotent, safe to re-fire." -Force | Out-Null

# --- Task 1b: the second (multi-round-trip) shadow watchdog, same shape ---
$action1b = New-ScheduledTaskAction -Execute $wsl `
    -Argument "-d Ubuntu -u root -e bash /root/gold-edge/ops/start_shadow_full_if_needed.sh"
$trigger1c = New-ScheduledTaskTrigger -AtLogOn
$trigger1d = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "GoldEdgeShadowFullWatchdog" -Action $action1b -Trigger @($trigger1c, $trigger1d) -Settings $settings1 -Principal $principal -Description "Keeps the Gold Edge shadow-full (multi-round-trip, no real orders) process running in WSL; idempotent, safe to re-fire." -Force | Out-Null

# --- Task 2: daily backfill, 04:10 + at logon catch-up ---
$action2 = New-ScheduledTaskAction -Execute $wsl `
    -Argument "-d Ubuntu -u root -e bash /root/gold-edge/ops/daily_backfill.sh"
$trigger2a = New-ScheduledTaskTrigger -Daily -At "04:10"
$trigger2b = New-ScheduledTaskTrigger -AtLogOn
$settings2 = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName "GoldEdgeDailyBackfill" -Action $action2 -Trigger @($trigger2a, $trigger2b) -Settings $settings2 -Principal $principal -Description "Re-runs the free keyless Kalshi/PAXG history backfill for Gold Edge daily (idempotent)." -Force | Out-Null

Get-ScheduledTask -TaskName "GoldEdgeShadowWatchdog", "GoldEdgeShadowFullWatchdog", "GoldEdgeDailyBackfill" | Select-Object TaskName, State
