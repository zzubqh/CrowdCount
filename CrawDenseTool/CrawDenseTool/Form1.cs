using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CrawDenseTool
{
    public partial class Form1 : Form
    {
        private string img_dir = @"E:\code\crowdcount\dataset\test";
        private static string res_dir = @"E:\code\crowdcount\result\dense_level\";
        private static int img_current_index = 0;
        private List<string> img_path = new List<string>();
        private List<ResNode> result = new List<ResNode>();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBox1.SelectedIndex = 0;
            PictureBoxInit();
        }

        private void PictureBoxInit()
        {
            DirectoryInfo TheFolder = new DirectoryInfo(img_dir);
            foreach (FileInfo NextFile in TheFolder.GetFiles())
            {
                string temp = NextFile.Name.ToLower();
                if (temp.Contains(".jpg") == false && temp.Contains(".png") == false)
                    continue;
                // 图片路径集初始化
                img_path.Add(Path.Combine(img_dir, NextFile.Name));

                // 图片结果集初始化
                ResNode res_node = new ResNode();
                res_node.imgName = NextFile.Name;
                res_node.crowCount = 0;
                result.Add(res_node);
            }            

            pictureBox1.Load(img_path[img_current_index]);
            this.Text = Path.GetFileName(img_path[img_current_index]);

            if (img_current_index == 0)
                button_next.Enabled = true;
            else if (img_current_index == img_path.Count - 1)
                button_pre.Enabled = true;
            else
            {
                button_next.Enabled = true;
                button_pre.Enabled = true;
            }
        }

        private void button_next_Click(object sender, EventArgs e)
        {
            try
            {
                // 记录结果
                string cur_level = comboBox1.SelectedItem.ToString();
                result[img_current_index].crowCount = int.Parse(cur_level);

                // 转到下一张
                img_current_index++;
                if (img_current_index >= img_path.Count)
                {
                    img_current_index = img_path.Count - 1;
                    MessageBox.Show("已经完工了，吼吼~ ^_^");
                }
                pictureBox1.Load(img_path[img_current_index]);
                this.Text = Path.GetFileName(img_path[img_current_index]);

                if (img_current_index == img_path.Count - 1)
                {
                    button_next.Enabled = false;
                }
                if (img_current_index == 1)
                {
                    button_pre.Enabled = true;
                }
            }
            catch (Exception ex)
            {
                //MessageBox.Show(ex.StackTrace, "哈哈，又报错啦！", MessageBoxButtons.OK);
            }
        }

        private void button_pre_Click(object sender, EventArgs e)
        {
            try
            {
                // 记录结果
                string cur_level = comboBox1.SelectedItem.ToString();
                result[img_current_index].crowCount = int.Parse(cur_level);

                // 转到上一张
                img_current_index--;
                if (img_current_index < 0)
                {
                    img_current_index = 0;
                    MessageBox.Show("都已经是第一张了，没法往前了:(");
                }
                pictureBox1.Load(img_path[img_current_index]);
                this.Text = Path.GetFileName(img_path[img_current_index]);                
                if (img_current_index == 0)
                {
                    button_pre.Enabled = false;
                }
                if (img_current_index == img_path.Count - 2)
                {
                    button_next.Enabled = true;
                }
            }
            catch (Exception ex)
            {
                //MessageBox.Show(ex.StackTrace, "哈哈，又报错啦！", MessageBoxButtons.OK); 
            }
        }


        private void button_save_Click(object sender, EventArgs e)
        {
            // 记录当前图片的结果
            string cur_level = comboBox1.SelectedItem.ToString();
            result[img_current_index].crowCount = int.Parse(cur_level);

            string file_path = res_dir + DateTime.Now.ToString("HHmmss") + ".csv";
            FileStream fs = new FileStream(file_path, FileMode.Create);
            StreamWriter sw = new StreamWriter(fs);
            //开始写入
            foreach(ResNode node in result)
            {
                sw.WriteLine(string.Format("{0},{1}", node.imgName, node.crowCount));
            }            
            //清空缓冲区
            sw.Flush();
            //关闭流
            sw.Close();
            fs.Close();
            MessageBox.Show("保存成功！");
        }

        private void Form1_KeyUp(object sender, KeyEventArgs e)
        {
            // A或a
            if (e.KeyValue == 65 || e.KeyValue == 97)
                button_pre_Click(sender, e);

            if (e.KeyValue == 68 || e.KeyValue == 100)
                button_next_Click(sender, e);

            if (e.Control && e.KeyCode == Keys.S)
                button_save_Click(sender, e);
        }
    }

    class ResNode
    {
        public string imgName;
        public int crowCount;
    }

}
