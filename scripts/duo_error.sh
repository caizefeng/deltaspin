grep "DAV" $1 | awk '{print $4}' > energy_error.txt
grep "Step" $1 | grep -E "\-\- 1\W" | awk '{print $11}' > mag_error.txt
grep "RMS:" $1 -A 1 | tail -1 >> mag_error.txt
