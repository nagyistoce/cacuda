package edu.lsu.cct.cacuda;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;


class KernelData {
	final String name;
	public KernelData(String name) {
		this.name = name;
	}	
	
	Map<String,String> attrs = new LinkedHashMap<String,String>();
	Map<String,VarData> vars = new LinkedHashMap<String,VarData>();
	Set<String> parameters = new HashSet<String>();
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CCTK_CUDA_KERNEL ");
		sb.append(name);
		sb.append("\n ATTRS:");
		for(Map.Entry<String,String> e : attrs.entrySet()) {
			sb.append(' ');
			sb.append(e.getKey());
			sb.append('=');
			sb.append('"');
			sb.append(e.getValue());
			sb.append('"');
		}
		sb.append("\n PARAMS:");
		for(String p : parameters) {
			sb.append(' ');
			sb.append(p);
		}
		sb.append("\n VARS:\n");
		for(Map.Entry<String, VarData> e : vars.entrySet()) {
			sb.append("  ");
			sb.append(e.getValue().toString());
		}
		return sb.toString();
	}
}