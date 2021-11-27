#!/bin/bash


LOCAL_PORT=8899
USERNAME=dvruette
HOST=euler.ethz.ch
WAITING_TIME_INTERVAL=30

# Cleanup
echo "Cleaning up ..."
ssh -T $USERNAME@$HOST <<ENDSSH
if [ -f \$HOME/VSCode_tunnel ]; then
  echo -e "Found old VSCode_tunnel file, deleting it ..."
  rm \$HOME/VSCode_tunnel
fi
ENDSSH

# Find old tunnel and delete it
pid=$(ps | grep ssh | grep $LOCAL_PORT | cut -d " " -f 1)
if [ ! -z "$pid" ]; then
  echo "Closing open tunnel on port $LOCAL_PORT"
  kill $pid
fi

ssh $USERNAME@$HOST bsub -J "code_server" -n 1 -W 12:00 -R "rusage[mem=2800]" <<ENDBSUB
module load gcc/6.3.0 code-server/3.9.3 eth_proxy
port=\$((3 * 2**14 + RANDOM % 2**14))
ip=\$(hostname -i)
echo "\${ip}:\${port}" > \$HOME/VSCode_tunnel
code-server --bind-addr=\${ip}:\${port}
ENDBSUB

ssh $USERNAME@$HOST "while ! [ -e \$HOME/VSCode_tunnel ]; do echo 'Waiting for code server to start, sleep for $WAITING_TIME_INTERVAL sec'; sleep $WAITING_TIME_INTERVAL; done"

remote_address=$(ssh $USERNAME@$HOST "cat \$HOME/VSCode_tunnel")
ssh -q -N -L localhost:$LOCAL_PORT:$remote_address $USERNAME@$HOST &

# SSH tunnel is started in the background, pause 5 seconds to make sure
# it is established before starting the browser
sleep 5

# save url in variable
appurl=http://localhost:$LOCAL_PORT
echo -e "Starting browser and connecting it to code server"
echo -e "Connecting to url "$appurl

APP=$(find "$HOME/Applications" -name "code-server.app" | head -n 1)
if [[ ! -z APP ]]; then
  open "$APP"
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
	xdg-open $appurl
elif [[ "$OSTYPE" == "darwin"* ]]; then
	open $appurl
else
	echo -e "Your operating system does not allow to start the browser automatically."
  echo -e "Please open $appurl in your browser."
fi