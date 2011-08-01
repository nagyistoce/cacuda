package edu.lsu.cct.cacuda;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.lsu.cct.piraha.Grammar;
import edu.lsu.cct.piraha.Group;
import edu.lsu.cct.piraha.Matcher;

public class CCLParser
{
  Grammar g = new Grammar();
  Matcher m = null;
  File src = null;
  private OutputGen out;
  private File templateDir;

  public CCLParser()
  {
    g.compile("w", "([ \t\r\n]|#.*)*");
    g.compile("w1", "([ \t\r\n]|#.*)+");
    g.compile("KERNEL",
        "CCTK_CUDA_KERNEL{-w1}{name}{-w1}({key}{-w}={-w}{value}{-w})*\\{{-w}({VAR}{-w}|{PAR}{-w})*\\}{-w}");
    g.compile("KERNELS", "^{-w}({KERNEL})*$");
    g.compile("name", "[A-Za-z0-9_]+");
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
  }

  public static void main(String[] args) throws Exception
  {
    // String str = Grammar.readContents(new
    // File("/home/sbrandt/GBSC12/code/CaCUDA/CaCUDACFD3D/cacuda.ccl"));
    CCLParser parser = new CCLParser();
    if (args.length != 2)
      usage();
    File cclFile = new File(args[0]);
    File templateDir = new File(args[1]);
    if (!cclFile.exists() || !templateDir.exists())
      usage();
    parser.parseFile(cclFile,templateDir);
  }

  public static void usage()
  {
    System.out.println("Usage: CCLParser cclFile templateDir");
    System.out.println("  where templateDir is a directory containing the templates");
    System.exit(1);
  }

  public void parseFile(File cclFile,File templateDir) throws IOException
  {
    if(!templateDir.exists() || !templateDir.isDirectory())
      usage();
    src = new File(cclFile.getParentFile(), "src");
    this.templateDir = templateDir;
    match(Grammar.readContents(cclFile));
  }

  Map<String, KernelData> kernels = new HashMap<String, KernelData>();

  public void match(String str) throws IOException
  {
    kernels = new HashMap<String, KernelData>();
    m = g.matcher("KERNELS", str);
    if (m.match(0))
    {
      processMatch();
      generateFiles();
    } else
      System.out.println("Error near line " + m.near());
  }

  private void generateFiles() throws IOException
  {
    for (KernelData kd : kernels.values())
    {
      generateFile(kd,"h");
    }
    generateFile(makeMasterKernel("Vars"),"h");
    generateFile(makeMasterKernel("Comm"),"cu");
  }

  private KernelData makeMasterKernel(String type)
  {
    KernelData master = new KernelData("master");
    master.attrs.put("TYPE", type);
    for (KernelData kd : kernels.values())
    {
      for (VarData vd : kd.vars.values())
      {
        master.vars.put(vd.name, new VarData(vd.name));
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

  private void generateFile(final KernelData kd,final String suffix) throws IOException
  {
    final String type = kd.attrs.get("TYPE");
    out = new OutputGen();
    File inFile = new File(templateDir,"cacuda."+type.toLowerCase()+"-template."+suffix);
    if(!inFile.exists()) {
      System.err.println("Error: file "+inFile.getAbsolutePath()+" does not exist");
      usage();
    }
    out.readSource(inFile);
    final File outFile = new File(src, "cctk_CaCUDA_" + kd.name + "_" + type + "."+suffix);
    try
    {
      System.out.println("Creating file " + outFile);
      System.out.println(" Using Data: " + kd);

      final int[] tile = parseArray(kd.attrs.get("TILE"), new int[]{ 16,16,16});
      final int[] stencil = parseArray(kd.attrs.get("STENCIL"), new int[]{ 1, 1, 1, 1, 1, 1 });
      
      //BufferedWriter bw = new BufferedWriter(fw);
      Object f = new CCLFormater(kd, tile, outFile, stencil, g);
      
      FileWriter fw = new FileWriter(outFile);
      BufferedWriter bw = new BufferedWriter(fw);
      DefConWriter dcw = new DefConWriter(bw);
      
      out.output(dcw, OutputGen.DEFAULT_TEMPLATE, f);
      
      //gen.readSource(br);
    } catch(IOException ioe) {
      throw new RuntimeException(ioe);
    }
  }

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
    // System.out.println(kd);
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
