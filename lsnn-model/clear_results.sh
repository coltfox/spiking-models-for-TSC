read -p "Are you sure you want to clear all results? [Y\N] " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -r ../results
fi