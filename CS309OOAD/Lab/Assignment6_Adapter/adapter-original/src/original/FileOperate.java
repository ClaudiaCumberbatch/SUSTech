package original;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FileOperate implements FileOperateInterfaceV1 {

    @Override
    public List<StaffModel> readStaffFile() {
        List<StaffModel> list = new ArrayList<>();
        try {
            URL url = new URL("https://cse.sustech.edu.cn/en/people/tenure/type/professor");
            Document doc = Jsoup.parse(url, 10000);
            Elements teachers = doc.getElementsByClass("teacher-box");
            teachers.forEach((element) -> {
                Element teacherRoot = element.getElementsByClass("text").first();
                StaffModel staffModel = new StaffModel();
                staffModel.setName(teacherRoot.child(0).text());
                staffModel.setTitle(teacherRoot.child(1).text());
                staffModel.setEmail(teacherRoot.child(2).text().replace("_AT_", "@"));
                staffModel.setRoom(teacherRoot.child(3).text());
                if (!teacherRoot.getElementsByClass("teacher-box-href").isEmpty())
                    staffModel.setLink(element.getElementsByClass("teacher-box-href").first().attr("href"));
                else {
                    staffModel.setLink("Haven't uploaded");
                }
                list.add(staffModel);
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("finish read");
        return list;
    }

    @Override
    public void printStaffFile(List<StaffModel> list) {
        if (list.isEmpty()) {
            System.out.println("no staff information");
        } else {
            for (StaffModel s : list) {
                System.out.println(s);
            }
        }

    }

    @Override
    public void writeStaffFile(List<StaffModel> list) {
        try {
            if (list.isEmpty()) {
                System.out.println("No information to be written");
                return;
            }
            //todo: change your file path
            FileWriter f = new FileWriter("staff.txt");
            BufferedWriter bufw = new BufferedWriter(f);
            String str = "";
            for (StaffModel s : list) {
                bufw.write(s.toString());
            }
            bufw.flush();
            bufw.close();
            System.out.println("finish writing");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
