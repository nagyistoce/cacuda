package edu.lsu.cct.cacuda;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.CharBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.lsu.cct.piraha.Grammar;
import edu.lsu.cct.piraha.Group;
import edu.lsu.cct.piraha.Matcher;

public class CCLParser
{
  Grammar g = new Grammar();
  Grammar g_deps = new Grammar();
  Matcher m = null;
  Matcher m_deps = null;
  File src = null;
  TemplateGroup root = null;
  private OutputGen out;
  private File templateDir;
  
  private final String fileSplitterRegex = "\\s*" +
  									  "############################################################\\s+" +
  									  "#CAKERNEL AUTO GENERATED PART. DO NOT EDIT BELOW THIS POINT#\\s+" +
  									  "############################################################";
  private final String fileSplitter = "\n" +
									  "############################################################\n" +
									  "#CAKERNEL AUTO GENERATED PART. DO NOT EDIT BELOW THIS POINT#\n" +
									  "############################################################\n";
 

  public CCLParser()
  {
    g.compile("w", "([ \t\r\n]|#.*)*");
    g.compile("w1", "([ \t\r\n]|#.*)+");
    g.compile("KERNEL",
        "CCTK_CUDA_KERNEL{-w1}{name}{-w1}({key}{-w}={-w}{value}{-w})*\\{{-w}({VAR}{-w}|{PAR}{-w})*\\}{-w}");
    g.compile("KERNELS", "^{-w}({KERNEL})*$");
    g.compile("name", "[A-Za-z0-9_/*]+");
    g.compile("key", "{name}");
    g.compile("value", "{name}|{dquote}|{squote}");
    g.compile("dquote", "\"(\\\\[^]|[^\\\\\"])*\"");
    g.compile("squote", "'(\\\\[^]|[^\\\\'])*'");
    g.compile("VAR",
        "CCTK_CUDA_KERNEL_VARIABLE({-w1}({key}{-w}={-w}{value}{-w})*|)\\{{-w}{name}({-w},{-w}{name})*{-w}\\}{-w}{dquote}");
    g.compile("PAR",
        "CCTK_CUDA_KERNEL_PARAMETER({-w1}({key}{-w}={-w}{value}{-w})*|)\\{{-w}{name}({-w},{-w}{name})*{-w}\\}{-w}{dquote}");
    g.compile("digit", "[0-9]+");
    g.compile("any","[^]*");
    g.compile("par","{key}{-w}={-w}{any}");
    
    /// The deps.ccl parse grammar. 
    g_deps.compile("w", "([ \t\r\n]|#.*)*");
    g_deps.compile("w1", "([ \t\r\n]|#.*)+");
    g_deps.compile("TEMPLATE_GROUP",
        "CCTK_KERNEL_TEMPLATE_GROUP{-w1}{name}{-w1}({key}{-w}={-w}{value}{-w})*\\{{-w}({TEMPLATE_GROUP}{-w}|{TEMPLATE}{-w}|{SCRIPT}{-w})*\\}{-w}");
    g_deps.compile("TEMPLATE_GROUPS", "^{-w}({TEMPLATE_GROUP})*$");
    g_deps.compile("name", "[A-Za-z0-9_/*$]+");
    g_deps.compile("key", "{name}");
    g_deps.compile("value", "{name}|{dquote}|{squote}");
    g_deps.compile("dquote", "\"(\\\\[^]|[^\\\\\"])*\"");
    g_deps.compile("squote", "'(\\\\[^]|[^\\\\'])*'");
    g_deps.compile("SCHEDULE",
    	"schedule{-w1}(({key}{-w}={-w}{value}{-w})*)\\{{-w}({value}({-w},{-w}{value})*{-w})?\\}{-w}{dquote}?");
    g_deps.compile("TRIGGERS",
		"triggers{-w1}(({key}{-w}={-w}{value}{-w})*)\\{{-w}({value}({-w},{-w}{value})*{-w})?\\}{-w}{dquote}?");
    g_deps.compile("TEMPLATE",
        "CCTK_KERNEL_TEMPLATE{-w1}{name}{-w1}({key}{-w}={-w}{value}{-w})*\\{{-w}(({TRIGGERS}|{SCHEDULE})({-w},{-w}{TRIGGERS}|{SCHEDULE})*{-w})?\\}{-w}{dquote}?");
    g_deps.compile("SCRIPT",
        "CCTK_KERNEL_SCRIPT{-w1}{name}{-w1}({key}{-w}={-w}{value}{-w})*\\{{-w}(({TRIGGERS}|{SCHEDULE})({-w},{-w}{TRIGGERS}|{SCHEDULE})*{-w})?\\}{-w}{dquote}?");
    g_deps.compile("digit", "[0-9]+");
    g_deps.compile("any","[^]*");
    g_deps.compile("par","{key}{-w}={-w}{any}");
  }

  public static void main(String[] args) throws Exception
  {
    // String str = Grammar.readContents(new
    // File("/home/sbrandt/GBSC12/code/CaCUDA/CaCUDACFD3D/cacuda.ccl"));
    CCLParser parser = new CCLParser();
    if (args.length != 2)
      usage();
    File cclFile = new File(args[0]);
    File depsCclFile = new File(args[1]);
    File templateDir = depsCclFile.getParentFile();
    if (!(cclFile.exists() && templateDir.exists() && depsCclFile.exists())) usage();
    parser.parseFiles(cclFile, depsCclFile, templateDir);
  }

  public static void usage()
  {
    System.err.println("Usage: CCLParser cclFile templateDir");
    System.err.println("  where templateDir is a directory containing the templates");
    System.exit(1);
  }

  public void parseFiles(File cclFile, File depsCclFile, File templateDir) throws Exception
  {
    src = new File(cclFile.getParentFile(), "src");
    this.templateDir = templateDir;
    root = new TemplateGroup("root");
    root.file = templateDir;
    match(Grammar.readContents(cclFile), Grammar.readContents(depsCclFile));
  }

  Map<String, KernelData> kernels = new HashMap<String, KernelData>();

  public void match(String str, String str_deps) throws Exception
  {
    kernels = new HashMap<String, KernelData>();
    m = g.matcher("KERNELS", str);
    m_deps = g_deps.matcher("TEMPLATE_GROUPS", str_deps);
    
    if (m_deps.match(0)) {
      root.parseInput(m_deps);
    } else throw new RuntimeException("Error in deps file, near line " + m_deps.near());
    
    if (m.match(0))
    {
      processMatch();
      generateFiles();
    } else throw new RuntimeException("Error near line " + m.near());
  }

  private void insertGroupScope(Map<String, Pair<LinkedHashSet<Node>, LinkedHashSet<KernelData>>> groupScope, Pair<Node, KernelData> p, String path){
	  String groupPath = path.replaceAll("/.*", "");
	  Pair<LinkedHashSet<Node>, LinkedHashSet<KernelData>> pair = groupScope.get(groupPath);
	  if(pair == null) pair = 
		  new Pair<LinkedHashSet<Node>, LinkedHashSet<KernelData>>(new LinkedHashSet<Node>(), new LinkedHashSet<KernelData>());
	  pair.first.add(p.first);
	  pair.second.add(p.second);
	  groupScope.put(groupPath, pair);
  }
  
  private void generateFiles() throws Exception
  {
	Map<String, Pair<LinkedHashSet<Node>, LinkedHashSet<KernelData>>> groupScope = 
		new LinkedHashMap<String, Pair<LinkedHashSet<Node>,LinkedHashSet<KernelData>>>();
	Map<String, Node> fileScope = 
		new LinkedHashMap<String, Node>();
	Map<String, Pair<Node, KernelData>>     evaluatedTemplates = 
		new LinkedHashMap<String, Pair<Node,KernelData> >();
	
    for (KernelData kd : kernels.values())
    {
    	LinkedHashMap<String, Node> pathToNodes = new LinkedHashMap<String, Node>();
    	String type = kd.attrs.get("TYPE");
    	root.findAll(type, "", pathToNodes);
    	if(pathToNodes.isEmpty())
    		throw new RuntimeException("Cannot find proper path to the template: " + type);
    	
    	for(Map.Entry<String, Node> entry : pathToNodes.entrySet()){
    		Template template = null;
    		try{ template = (Template)(entry.getValue());}
    		catch(Exception e){ throw new RuntimeException("Only template files can be pointed in CCL file so far.");}
    		
    		if(entry.getValue().scope == Node.Scope.group)
    			insertGroupScope(groupScope, new Pair<Node, KernelData>(entry.getValue(), kd), entry.getKey());
    		else
    			fileScope.put(entry.getKey(), entry.getValue());    			
    		
    		int len = fileScope.size(), newlen = 0;
    		
    		while (newlen != len){
    			len = fileScope.size();
    			for (Map.Entry<String, Node> entry2 : fileScope.entrySet()){
    				Node n = entry2.getValue();    				
    				for(Node tmpn : n.triggers.values()){
    					if(tmpn.scope == Node.Scope.group)
    		    			insertGroupScope(groupScope, new Pair<Node, KernelData>(tmpn, kd), entry.getKey());
    		    		else
    		    			fileScope.put(tmpn.abspath, tmpn); 
    				}
    			}
    			newlen = fileScope.size();    			
    		}
    		
    		Iterator<Map.Entry<String, Node>> it = fileScope.entrySet().iterator();
    		while(it.hasNext()){
    			Map.Entry<String, Node> entry2 = it.next();
    			Node n = entry2.getValue();
    			Pair<String, String> filePaths = ((Template)n).produceFilenames(kd);
    			//since one template may be evaluated for several 
    			evaluatedTemplates.put(filePaths.second, new Pair<Node, KernelData>(n, kd));
    			generateFile(kd, filePaths.first, filePaths.second, (Template)n, !"no".equals(n.attrs.get("replace")));
    			it.remove();
    		}
    	}    	
    }
    
    
    //evaluete group templates; eithere scheduled or triggered
	for (Map.Entry<String, Pair<LinkedHashSet<Node>, LinkedHashSet<KernelData>>> entry : groupScope.entrySet()){
		KernelData groupkd = new KernelData(entry.getKey());
		for (KernelData kd : entry.getValue().second)
	    {
	      for (VarData vd : kd.vars.values())
	      {
	    	if (groupkd.vars.containsKey(vd.name)){
	    		VarData vdtmp = groupkd.vars.get(vd.name);
	    		if(!"separateinout".equals(vdtmp.attrs.get("intent")))
	    			groupkd.vars.put(vd.name, vd);
	    	}else groupkd.vars.put(vd.name, vd);
	      }
	    }
		for (Node n : entry.getValue().first){
			Pair<String, String> filePaths = ((Template)n).produceFilenames(groupkd, true);
			generateFile(groupkd, filePaths.first, filePaths.second, (Template)n, !"no".equals(n.attrs.get("replace")));
			evaluatedTemplates.put(filePaths.second, new Pair<Node, KernelData>(n, groupkd));
		}
	}    
	
	//generate schedule & compilation
	File scheduleFile = new File(src.getParentFile(), "schedule.ccl");
	if(!scheduleFile.exists()) throw new RuntimeException("Cannot find schedule file: " + scheduleFile.getAbsolutePath());
	File makeDefnFile = new File(src, "make.code.defn");
	if(!makeDefnFile.exists()) throw new RuntimeException("Cannot find make.code.defn file: " + scheduleFile.getAbsolutePath());
	
	StringBuffer scheduleBuffer = new StringBuffer();
	StringBuffer makeDefnBuffer = new StringBuffer();
	CharBuffer tmpbuff =  CharBuffer.allocate((int)scheduleFile.length() + 1);
	FileReader tmpreader = new FileReader(scheduleFile);tmpreader.read(tmpbuff); tmpreader.close();
	scheduleBuffer.append(tmpbuff.rewind().toString().split(fileSplitterRegex)[0]).append("\n").append(fileSplitter).append("\n");
	
	tmpbuff = CharBuffer.allocate((int)makeDefnFile.length() + 1); 
	tmpreader = new FileReader(makeDefnFile); tmpreader.read(tmpbuff); tmpreader.close(); 
	makeDefnBuffer.append(tmpbuff.rewind().toString().split(fileSplitterRegex)[0]).append("\n").append(fileSplitter).append("\nSRCS += ");
	
	for (Map.Entry<String, Pair<Node, KernelData>> entry : evaluatedTemplates.entrySet()){
		String f = entry.getKey();
		Node n = entry.getValue().first;
		
		for (String s : n.schedule){
			if('"' == s.charAt(0) && '"' == s.charAt(s.length() - 1))
				s = s.substring(1, s.length() - 1).replaceAll("\\\\[\"]", "\"");
			if('\'' == s.charAt(0) && '\'' == s.charAt(s.length() - 1))
				s = s.substring(1, s.length() - 1).replaceAll("\\\\[']", "\'");
			scheduleBuffer.append(s).append("\n");
		}
		
		if("yes".equals(n.attrs.get("compile")))
			makeDefnBuffer.append(f).append(" ");
	}
	
	FileWriter tmpWriter;
	(tmpWriter = new FileWriter(scheduleFile)).write(scheduleBuffer.append("\n").toString()); tmpWriter.close();
	(tmpWriter = new FileWriter(makeDefnFile)).write(makeDefnBuffer.append("\n").toString()); tmpWriter.close();	
  }

  private KernelData makeMasterKernel(String type)
  {
    KernelData master = new KernelData("master");
    master.attrs.put("TYPE", type);
    for (KernelData kd : kernels.values())
    {
      for (VarData vd : kd.vars.values())
      {
    	if (master.vars.containsKey(vd.name)){
    		VarData vdtmp = master.vars.get(vd.name);
    		if(!"separateinout".equals(vdtmp.attrs.get("intent"))){
    			master.vars.put(vd.name, vd);
    		}    			
    	}else master.vars.put(vd.name, vd);
      }
    }
    return master;
  }

  private int[] parseArray(String in, int[] def)
  {
    if (in == null)
      return def;
    Matcher m = g.matcher("digit", in);
    int pos = 0;
    int[] array = new int[def.length];
    int index = 0;

    while (m.find(pos))
    {
      array[index++] = Integer.parseInt(m.substring());
      pos = m.getEnd();
    }
    return array;
  }

  private void generateFile(final KernelData kd, final String templatePath, final String dst, Template template) throws IOException{
	  generateFile(kd, templatePath, dst, template, true);
  }
  private void generateFile(final KernelData kd, final String templatePath, final String dst, Template template, boolean truncate) throws IOException
  {
    final String type = kd.attrs.get("TYPE");
    out = new OutputGen();
    File inFile = new File(templatePath);
    if(!inFile.exists()) {
      System.err.println("Error: file "+inFile.getAbsolutePath()+" does not exist");
      usage();
    }
    out.readSource(inFile);
    final File outFile = new File(src, dst);
    if(!truncate && outFile.exists()){
    	System.err.println("File " + outFile + " because already exists, and is set to not be replaced.");
    	return;
    }
    try
    {
      System.out.println("Creating file " + outFile);
      System.out.println(" Using Data: " + kd);

      final int[] tile = parseArray(kd.attrs.get("TILE"), new int[]{ 16,16,16});
      final int[] stencil = parseArray(kd.attrs.get("STENCIL"), new int[]{ 1, 1, 1, 1, 1, 1 });
      
      //BufferedWriter bw = new BufferedWriter(fw);
      Object f = new CCLFormater(kd, tile, outFile, stencil, g, template);
      
      FileWriter fw = new FileWriter(outFile);
      BufferedWriter bw = new BufferedWriter(fw);
      DefConWriter dcw = new DefConWriter(bw);
      
      out.output(dcw, OutputGen.DEFAULT_TEMPLATE, f);
      
      //gen.readSource(br);
    } catch(IOException ioe) {
      throw new RuntimeException(ioe);
    }
  }

//  void processMatch_deps()
//  {
//    for (int i = 0; i < m_deps.groupCount(); i++)
//      root.parseInput(m_deps.group(i));
//  }
  
  void processMatch()
  {
    for (int i = 0; i < m.groupCount(); i++)
    {
      processKernel(m.group(i));
    }
  }

  private void processKernel(Group m)
  {
    KernelData kd = null;

    for (int i = 0; i < m.groupCount(); i++)
    {
      String p = m.group(i).getPatternName();

      if ("name".equals(p))
      {
        kd = new KernelData(m.group(i).substring());
      } else if ("key".equals(p))
      {
        kd.attrs.put(m.group(i).substring(), m.group(i + 1).substring());
      } else if ("PAR".equals(p))
      {
        processPar(kd, m.group(i));
      } else if ("VAR".equals(p))
      {
        processVar(kd, m.group(i));
      }
    }
    kernels.put(kd.name, kd);
  }

  private void processVar(KernelData kd, Group m)
  {
    Map<String, String> attrs = new HashMap<String, String>();
    for (int i = 0; i < m.groupCount(); i++)
    {
      String p = m.group(i).getPatternName();

      if ("key".equals(p))
      {
        attrs.put(m.group(i).substring(), m.group(i + 1).substring());
      } else if ("name".equals(p))
      {
        VarData vd = new VarData(m.group(i).substring());

        vd.attrs = attrs;
        kd.vars.put(vd.name, vd);
      }
    }
  }

  private void processPar(KernelData kd, Group m)
  {
    for (int i = 0; i < m.groupCount(); i++)
    {
      if (m.group(i).getPatternName().equals("name"))
      {
        kd.parameters.add(m.group(i).substring());
      }
    }
  }
}
