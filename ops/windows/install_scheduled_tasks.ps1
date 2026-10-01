# Run this once, in a normal (non-admin) PowerShell window, on the machine
# that hosts the Ubuntu WSL distro at /root/gold-edge.
#
# Registers two per-user Scheduled Tasks so the Gold Edge shadow (no real
# orders, never places trades) and the daily free/keyless history backfill
# keep running after a reboot without anyone needing to notice and restart
# them by hand:
#
#   GoldEdgeShadowWatchdog  - fires at logon, then every 15 minutes forever.
#                             Calls ops/start_shadow_if_needed.sh, which is
#                             safe to call repeatedly: shadow_supervisor.sh
#                             takes an exclusive flock as its first action,
#                             so a duplicate launch just exits immediately
#                             instead of running two shadow processes
#                             against the same sqlite file.
#   GoldEdgeDailyBackfill   - fires daily at 04:10, plus once at logon as a
#                             catch-up if the PC was off at 04:10. Calls
#                             ops/daily_backfill.sh (idempotent: INSERT OR
#                             IGNORE), since Kalshi's own public API only
#                             retains ~2 months of settled-market history.
#
# Neither task needs admin rights to register (LogonType Interactive,
# RunLevel Limited, current user only) or to run.

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

# --- Task 2: daily backfill, 04:10 + at logon catch-up ---
$action2 = New-ScheduledTaskAction -Execute $wsl `
    -Argument "-d Ubuntu -u root -e bash /root/gold-edge/ops/daily_backfill.sh"
$trigger2a = New-ScheduledTaskTrigger -Daily -At "04:10"
$trigger2b = New-ScheduledTaskTrigger -AtLogOn
$settings2 = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName "GoldEdgeDailyBackfill" -Action $action2 -Trigger @($trigger2a, $trigger2b) -Settings $settings2 -Principal $principal -Description "Re-runs the free keyless Kalshi/PAXG history backfill for Gold Edge daily (idempotent)." -Force | Out-Null

Get-ScheduledTask -TaskName "GoldEdgeShadowWatchdog", "GoldEdgeDailyBackfill" | Select-Object TaskName, State
