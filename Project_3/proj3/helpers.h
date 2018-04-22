void log(std::string filename, std::string s){
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app);
	if(outfile.good()){
		outfile << s << std::endl;
	}
	outfile.close();
}
