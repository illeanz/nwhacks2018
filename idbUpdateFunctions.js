pdb.find().toArray(function(err, documents) {
	res.json(documents)
};


//////////////////////////////////////////////////////////////////////

var idb;

mydb2 = db.db('individual_db');
idb = mydb2.collection('individual_db');

app.get('/', (req, res) => {

	//Extract last 5 entries from individual_db
	inf_entries = idb.find().limit(5).toArray();

	//Use child_process to run a Python script --> does inference based on last 5 entries in individual DB
	var spawn = require("child_process").spawn;
	var process = spawn('python', ["path/to/script.py", "infer", inf_entries, json_path, model_path]);

	//Listen for output from python script --> if output is not null, generate new DB entry
	process.stdout.on('data', function (data) {
		console.log(data.toString());    
		res.write(data);
    	res.end('end');
	});
});
