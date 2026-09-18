#!/usr/bin/env python3
"""
AXUV Backpropagation Script
===========================
Re-runs AXUV analysis on historical shots and saves derived quantities to MDSPlus.

Usage:
  python3 axuv_backprop.py --shot-list 260530050 260525000 260520000
  python3 axuv_backprop.py --shot-range 260530000 260520000 100  # Start, end, step
  python3 axuv_backprop.py --shot-file shots.txt
"""

import subprocess
import sys
import argparse
import MDSplus as mds

def run_shot(shotnum, write_mds=True):
    """Run AXUV analysis on a single shot."""
    # Use the SAME interpreter that launched this script (sys.executable) rather
    # than a bare 'python3' from PATH — otherwise the child can resolve to an env
    # without MDSplus/PyAbel (e.g. anaconda) and fail on import.
    cmd = [sys.executable, '/home/whamdata/post_processing/axuv.py',
           '-s', str(shotnum), '--write-mds', str(write_mds)]
    
    print(f"\n{'='*60}")
    print(f"Processing shot {shotnum}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"❌ FAILED: axuv.py returned code {result.returncode}")
            print("STDERR:", result.stderr[:500])
            return False
        
        print("✓ axuv.py completed")
        
        # Verify data was saved
        if verify_mdsplus_save(shotnum):
            print(f"✓ Shot {shotnum} backpropped successfully")
            return True
        else:
            print(f"⚠ Warning: axuv.py ran but MDSPlus save may have failed")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: Shot {shotnum} took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def verify_mdsplus_save(shotnum):
    """Check if ANALYSIS nodes were saved to MDSPlus for this shot."""
    try:
        t = mds.Tree('wham', shotnum)
        found = 0
        for da in ['DIODEARRAY1', 'DIODEARRAY2', 'DIODEARRAY3']:
            try:
                node = t.getNode(f'DIAG.AXUV.{da}.ANALYSIS.CENTROID_HR')
                data = node.getData().data()
                print(f"  ✓ {da}: {len(data)} centroid samples saved")
                found += 1
            except:
                pass
        
        return found > 0
    except Exception as e:
        print(f"  ⚠ Could not verify MDSPlus: {str(e)[:60]}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--shot-list', nargs='+', type=int, 
                       help='Shot numbers to process')
    group.add_argument('--shot-range', nargs=3, type=int, metavar=('START', 'END', 'STEP'),
                       help='Shot range: start end step')
    group.add_argument('--shot-file', type=str,
                       help='File with one shot per line')
    
    parser.add_argument('--write-mds', type=bool, default=True, nargs='?', const=True,
                        help='Write to MDSPlus (default: True)')
    
    args = parser.parse_args()
    
    # Parse shot list
    shots = []
    if args.shot_list:
        shots = args.shot_list
    elif args.shot_range:
        start, end, step = args.shot_range
        shots = list(range(start, end - 1, -step))  # Reverse order (newest first)
    elif args.shot_file:
        with open(args.shot_file) as f:
            shots = [int(line.strip()) for line in f if line.strip()]
    
    print(f"\n{'='*60}")
    print(f"AXUV Backpropagation")
    print(f"{'='*60}")
    print(f"Shots to process: {len(shots)}")
    print(f"First 5: {shots[:5]}")
    print(f"{'='*60}\n")
    
    # Run backprop
    results = {}
    for i, shot in enumerate(shots, 1):
        print(f"\n[{i}/{len(shots)}]", end=" ")
        success = run_shot(shot, write_mds=args.write_mds)
        results[shot] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    succeeded = sum(1 for v in results.values() if v is True)
    warned = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    print(f"✓ Succeeded: {succeeded}")
    print(f"⚠ Warned:   {warned}")
    print(f"❌ Failed:   {failed}")
    
    if failed == 0:
        print("\n✓ Backprop complete!")
        sys.exit(0)
    else:
        print(f"\n❌ {failed} shots failed. Review output above.")
        sys.exit(1)
